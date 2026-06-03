import os
import json
import time
from pathlib import Path
from pymongo import MongoClient

class StorageManager:
    """
    Handles storage and status reporting for the Kairos pipeline.
    Supports both local filesystem (checkpoint.json) and remote MongoDB.
    """

    def __init__(self, chat_id=None, mongo_uri=None, local_path=None, video_name=None):
        self.chat_id = chat_id
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI")
        self.local_path = Path(local_path) if local_path else None
        self.video_name = video_name
        
        # Don't reset client/db if we're just updating the video/chat context
        if not hasattr(self, 'client'): 
            self.client = None
            self.db = None
        self.is_remote = False

        # AUTO-SYNC: If we have a URI, we enable remote mode automatically
        if self.mongo_uri:
            # If no chat_id provided, we generate a deterministic fallback ID
            # based on the video name so results stay grouped in MongoDB.
            if not self.chat_id and self.video_name:
                import hashlib
                clean_name = Path(self.video_name).stem.replace(" ", "_")
                # Create a 24-character hex string (valid ObjectId length) from md5
                m = hashlib.md5(clean_name.encode()).hexdigest()[:24]
                self.chat_id = m
                print(f"[StorageManager] No chat_id provided. Using deterministic native ID: {self.chat_id}")
            
            if self.chat_id:
                try:
                    if not self.client:
                        self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
                        # Verify connection
                        self.client.admin.command('ping')
                    
                    # Get DB name from environment, then URI, then default to "kairos"
                    try:
                        db_name = os.getenv("MONGODB_DB_NAME") or self.client.get_database().name
                    except Exception:
                        db_name = "test"
                    
                    self.db = self.client[db_name]
                    self.is_remote = True
                    # Only print on first connection
                    if not hasattr(self, '_connected_printed'):
                        print(f"[StorageManager] Connected to MongoDB: {db_name}")
                        self._connected_printed = True
                except Exception as e:
                    print(f"[StorageManager] WARNING: Failed to connect to MongoDB: {e}")
                    self.is_remote = False

    @staticmethod
    def _utc_now():
        """Returns the current UTC time as an ISO-like string used across Mongo writes."""
        return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    def _base_chat_on_insert(self):
        """Provides only insert-only defaults so upserts never conflict with live $set fields."""
        now = self._utc_now()
        return {
            "messages": [],
            "messageCount": 0,
            "deletedAt": None,
            "createdAt": now,
            "lastMessageAt": None,
        }

    @staticmethod
    def _normalize_objects(raw_objects):
        """
        Normalizes model object detections into the string array expected by chat_chunks.objects.
        """
        if not isinstance(raw_objects, list):
            return []

        normalized = []
        for item in raw_objects:
            if isinstance(item, str):
                value = item.strip()
                if value:
                    normalized.append(value)
                continue

            if isinstance(item, dict):
                for key in ("label", "object", "name", "class"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        normalized.append(value.strip())
                        break

        return normalized

    @staticmethod
    def _parse_meta_context(context_value):
        """
        Parses a synopsis-level embedding context into one of the supported typed meta chunks.
        """
        if not isinstance(context_value, str):
            return None, None

        raw_value = context_value.strip()
        if not raw_value or ": " not in raw_value:
            return None, None

        chunk_type, payload = raw_value.split(": ", 1)
        chunk_type = chunk_type.strip()
        payload = payload.strip()
        allowed_types = {"summary", "highlights", "timeline", "suggested_clips", "questions"}

        if chunk_type not in allowed_types or not payload:
            return None, None

        return chunk_type, payload

    def update_pipeline_state(self, stage, percent, status="processing"):
        """Updates the current pipeline status in the chats collection."""
        if not self.is_remote:
            return

        try:
            now = self._utc_now()
            update_data = {
                "pipeline.lastStage": stage,
                "pipeline.percent": percent,
                "pipeline.state": status,
                "pipeline.updatedAt": now,
                "updatedAt": now
            }
            
            # We use upsert=True so that if the chat doesn't exist (e.g. raw run), it gets created
            self.db.chats.update_one(
                {"_id": self._to_oid(self.chat_id)},
                {
                    "$set": update_data,
                    "$setOnInsert": self._base_chat_on_insert(),
                },
                upsert=True
            )
        except Exception as e:
            print(f"[StorageManager] ERROR updating pipeline state: {e}")

    def save_checkpoint(self, checkpoint):
        """Saves checkpoint data locally and optionally updates remote progress."""
        if self.local_path:
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.local_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        # In remote mode, we also update the last stage
        if self.is_remote and "steps" in checkpoint:
            # Infer stage from the last step in checkpoint steps
            last_step_name = list(checkpoint["steps"].keys())[-1] if checkpoint["steps"] else "initializing"
            # In a real scenario, we might want to map step names to display-friendly stages
            self.update_pipeline_state(stage=last_step_name, percent=None)

    def read_checkpoint(self):
        """Loads checkpoint from local filesystem."""
        if self.local_path and self.local_path.exists():
            try:
                with open(self.local_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[StorageManager] ERROR reading checkpoint: {e}")
        return {}

    def save_final_results(self, checkpoint, rag_embedding=None):
        """
        Saves final results to MongoDB:
        1. Replaces chat_chunks with v3 retrieval chunks for the chat.
        2. Updates the chat document with the generated summary and marks the pipeline as ready.
        """
        if not self.is_remote:
            # In local mode, make_embedding_log already saves rag_embedding.json
            return

        try:
            now = self._utc_now()
            chat_oid = self._to_oid(self.chat_id)

            # 1. Prepare chat_chunks
            chunks = []
            scenes = checkpoint.get("scenes", [])
            
            # Decide how to access contexts/embeddings (handle full dict or just list)
            embed_list = []
            context_list = []
            if isinstance(rag_embedding, dict):
                embed_list = rag_embedding.get("embeddings", [])
                context_list = rag_embedding.get("contexts", [])
            elif isinstance(rag_embedding, list):
                embed_list = rag_embedding
            
            # Fill scene retrieval chunks.
            for i, scene in enumerate(scenes):
                embedding = None
                if i < len(embed_list):
                    embedding = embed_list[i]

                context = ""
                if i < len(context_list) and isinstance(context_list[i], str) and context_list[i].strip():
                    context = context_list[i].strip()
                else:
                    context = (scene.get("llm_scene_description") or "").strip()

                chunk = {
                    "chatId": chat_oid,
                    "chunkIndex": i,
                    "sceneIndex": i,
                    "type": "scene",
                    "startSec": scene.get("start_seconds"), 
                    "endSec": scene.get("end_seconds"),
                    "startTimecode": scene.get("start_timecode"),
                    "endTimecode": scene.get("end_timecode"),
                    "context": context,
                    "captions": scene.get("frame_captions", []),
                    "objects": self._normalize_objects(scene.get("yolo_detections", [])),
                    "audioSpeech": scene.get("audio_speech", ""),
                    "audioNatural": scene.get("audio_natural", ""),
                    "embedding": embedding,
                    "createdAt": now
                }
                chunks.append(chunk)

            # Fill one typed meta chunk per synopsis context emitted into rag embeddings.
            num_scenes = len(scenes)
            for j in range(num_scenes, len(context_list)):
                chunk_type, payload = self._parse_meta_context(context_list[j])
                if not chunk_type:
                    continue

                embedding = embed_list[j] if j < len(embed_list) else None
                meta_chunk = {
                    "chatId": chat_oid,
                    "chunkIndex": j,
                    "sceneIndex": None,
                    "type": chunk_type,
                    "context": payload,
                    "embedding": embedding,
                    "createdAt": now,
                    chunk_type: payload,
                }
                chunks.append(meta_chunk)

            if chunks:
                # Delete existing chunks for this chat to avoid duplicates on re-run
                self.db.chat_chunks.delete_many({"chatId": chat_oid})
                self.db.chat_chunks.insert_many(chunks)
                print(f"[StorageManager] Inserted {len(chunks)} chunks into MongoDB.")
            else:
                self.db.chat_chunks.delete_many({"chatId": chat_oid})
                print(f"[StorageManager] No scene chunks found for chat {self.chat_id}; existing chunks cleared.")

            # 2. Update chat document with the app-compatible summary object
            synopsis_data = checkpoint.get("synopsis", {})
            synopsis_payload = {
                "summary": "",
                "video_highlights": [],
                "video_timeline": [],
                "suggested_clips": [],
            }
            if isinstance(synopsis_data, dict):
                if isinstance(synopsis_data.get("summary"), str):
                    synopsis_payload["summary"] = synopsis_data.get("summary", "")
                for key in ("video_highlights", "video_timeline", "suggested_clips"):
                    value = synopsis_data.get(key)
                    if isinstance(value, list):
                        synopsis_payload[key] = value
            elif isinstance(synopsis_data, str):
                synopsis_payload["summary"] = synopsis_data

            video_stem = Path(self.video_name).stem if self.video_name else str(self.chat_id)
            summary_payload = {
                "title": f"Video Summary: {video_stem}",
                "videoName": video_stem,
                "synopsis": synopsis_payload,
                "generatedAt": now,
            }

            update_payload = {
                "pipeline.state": "ready",
                "pipeline.lastStage": "embedding",
                "pipeline.percent": 100,
                "pipeline.lastError": None,
                "pipeline.updatedAt": now,
                "summary": summary_payload,
                "updatedAt": now
            }
            
            self.db.chats.update_one(
                {"_id": chat_oid},
                {
                    "$set": update_payload,
                    "$setOnInsert": self._base_chat_on_insert(),
                },
                upsert=True
            )
            print(f"[StorageManager] Pipeline marked as READY for chat {self.chat_id}")

        except Exception as e:
            print(f"[StorageManager] ERROR saving final results: {e}")

    def _to_oid(self, id_str):
        """Helper to convert string ID to MongoDB ObjectId if possible."""
        from bson import ObjectId
        try:
            return ObjectId(id_str)
        except:
            return id_str # Return as string if not a valid ObjectId (e.g. mock IDs)

    @staticmethod
    def map_stage(step_name):
        """Maps an internal step name to a user-friendly display stage."""
        mapping = {
            "get_scene_list": "scene_detection",
            "save_clips": "clip_extraction",
            "sample_frames": "frame_sampling",
            "caption_frames": "frame_captioning",
            "sample_fps": "motion_sampling",
            "detect_object_yolo": "object_detection",
            "scan_audio": "audio_prescan",
            "extract_speech": "speech_transcription",
            "extract_sounds": "sound_analysis",
            "describe_scenes": "scene_description",
            "kg_extract": "knowledge_graph",
            "summarize_scenes": "narrative_synthesis",
            "synthesize_synopsis": "synopsis_generation",
            "make_embedding": "embedding"
        }
        return mapping.get(step_name, step_name)
