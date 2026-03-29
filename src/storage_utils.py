import os
import json
import time
from pathlib import Path
from pymongo import MongoClient, UpdateOne, InsertOne

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
                    self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
                    # Verify connection
                    self.client.admin.command('ping')
                    
                    # Get DB name from environment, then URI, then default to "kairos"
                    try:
                        db_name = os.getenv("MONGODB_DB_NAME") or self.client.get_database().name
                    except Exception:
                        db_name = "kairos"
                    
                    self.db = self.client[db_name]
                    self.is_remote = True
                    print(f"[StorageManager] Connected to MongoDB: {db_name}")
                except Exception as e:
                    print(f"[StorageManager] WARNING: Failed to connect to MongoDB: {e}")
                    self.is_remote = False

    def update_pipeline_state(self, stage, percent, status="processing"):
        """Updates the current pipeline status in the chats collection."""
        if not self.is_remote:
            return

        try:
            update_data = {
                "pipeline.lastStage": stage,
                "pipeline.percent": percent,
                "pipeline.state": status,
                "pipeline.updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }
            
            # We use upsert=True so that if the chat doesn't exist (e.g. raw run), it gets created
            self.db.chats.update_one(
                {"_id": self._to_oid(self.chat_id)},
                {"$set": update_data},
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
        1. Bulk inserts chat_chunks (metadata + embeddings).
        2. Updates the chat document with the synopsis and sets state to 'ready'.
        """
        if not self.is_remote:
            # In local mode, make_embedding_log already saves rag_embedding.json
            return

        try:
            # 1. Prepare chat_chunks
            chunks = []
            scenes = checkpoint.get("scenes", [])
            for i, scene in enumerate(scenes):
                # Retrieve embedding for this scene if available
                embedding = None
                if isinstance(rag_embedding, list) and i < len(rag_embedding):
                    embedding = rag_embedding[i]

                chunk = {
                    "chatId": self._to_oid(self.chat_id),
                    "chunkIndex": i,
                    "sceneIndex": i, # Assuming 1:1 for now
                    "startSec": scene.get("start_sec"),
                    "endSec": scene.get("end_sec"),
                    "startTimecode": scene.get("start_timecode"),
                    "endTimecode": scene.get("end_timecode"),
                    "context": scene.get("llm_scene_description", ""), # Merged context
                    "captions": scene.get("frame_captions", []),
                    "objects": scene.get("yolo_detections", []),
                    "audioSpeech": scene.get("audio_speech", ""),
                    "audioNatural": scene.get("audio_natural", ""),
                    "embedding": embedding,
                    "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
                }
                chunks.append(chunk)

            if chunks:
                # Delete existing chunks for this chat to avoid duplicates on re-run
                self.db.chat_chunks.delete_many({"chatId": self._to_oid(self.chat_id)})
                self.db.chat_chunks.insert_many(chunks)
                print(f"[StorageManager] Inserted {len(chunks)} chunks into MongoDB.")

            # 2. Update chat document with synopsis
            synopsis_data = checkpoint.get("synopsis", {})
            update_payload = {
                "pipeline.state": "ready",
                "pipeline.lastStage": "complete",
                "pipeline.percent": 100,
                "pipeline.updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "synopsis": synopsis_data,
                "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }
            
            # Use upsert=True to create the chat document if it doesn't exist
            self.db.chats.update_one(
                {"_id": self._to_oid(self.chat_id)},
                {"$set": update_payload},
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
            "summarize_scenes": "narrative_synthesis",
            "synthesize_synopsis": "synopsis_generation",
            "make_embedding": "embedding"
        }
        return mapping.get(step_name, step_name)
