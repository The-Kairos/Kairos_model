"""Mapping of video base-names to lists of test queries for retrieval comparison.

Keys match the video filename (without extension) found in processed checkpoints.
Each value is a list of natural-language queries used to evaluate retrieval quality.
"""

from typing import Dict, List

# Placeholder test queries map. Fill this in with the video base-names
# (filename without extension) as keys and a list of queries as values.
# Example keys match the `video_path` base name found in processed checkpoints.

TEST_QUERIES_MAP: Dict[str, List[str]] = {
    "Argentina v France Full Penalty Shoot-out": ["Give me the clip where Messi scores", 
                                                  "Where the Argentinian goalkeeper Martinez blocks the goal",
                                                  "Commentators speaking about Kylian Mbappe failing at the European championships against Switzerland",
                                                  "The Argentinian team celebrating and hugging on their victory",
                                                  "Retrieve the scenes where a crowd is shown"],
    "How to Make Pasta - Without a Machine": ["the scene where the person washes their hands",
                                              "the scene where they are rolling dough",
                                              "the scene where flour is being measured",
                                              "when they serve pasta",
                                              "a cooking demonstration"],
    # "NEW YORK TIMES SQUARE 2024 _ 4K WALK TOUR MORNING": ["The guy putting on a yellow costume",
    #                                                     "A guy wearing a captain america costume",
    #                                                     "The clip of the yellow/orange car passing",
    #                                                     "The clip of the fedex car passing",
    #                                                     "Two guys wearing red hats"],
    # "Statistical Learning_ 5.2 K-fold Cross Validation": ["The clip where K-fold cross-validation is defined",
    #                                                       "A clip showing the presenter",
    #                                                       "What is the formula of cross validation given fold K",
    #                                                       "The part where he explained the issues of cross validation",
    #                                                       "where is the importance of K-fold was explained?"],
    "Watch Malala Yousafzai's Nobel Peace Prize acceptance speech": ["Give me the clip of the woman with a colorful hijab sitting next to a man",
                                                                    "Give me the clip of Kailash Satyarthi wearing glasses and white clothes clapping for Malala",
                                                                    "Give me the clip of a room full of people clapping",
                                                                    "Where Malala says what her brothers call her",
                                                                    "Give me the clip where Malala fixes her pink hijab"],
    "Young Sheldon - First Day of High School": ["Give me the clip of the boy entering his class",
                                                "Give me the scene of the mom worried",
                                                "Show me the scenes that have music",
                                                "Show me clips of the school",
                                                "A clip where students are in a classroom"],
}

DEFAULT_NUM_QUERIES_PER_VIDEO = None  # leave to use list length; can be overridden by runner
