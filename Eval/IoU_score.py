from answer_evaluator import AnswerEvaluator
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

class IoUScore(AnswerEvaluator):

    def __init__(self):
        self.scoreList: list = None
    
    def set_setAnswer(self, setAnswer: str):
        return super().set_setAnswer(setAnswer)
    
    def set_genAnswer(self, genAnswer: str):
        return super().set_genAnswer(genAnswer)
    
    def find_longest_common_substring(text1, text2):
        """
        Identify the longest common word-level substring between two texts and return their spans.

        Args:
        text1 (str): The first text.
        text2 (str): The second text.

        Returns:
        tuple: A tuple containing the span for text1, the span for text2, and the longest common substring.
        """
        # Tokenize both texts
        words1 = word_tokenize(text1)
        words2 = word_tokenize(text2)

        # Length of both tokenized texts
        len1, len2 = len(words1), len(words2)

        # Create a matrix to store lengths of longest common suffixes
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        longest = 0  # Length of the longest common substring
        end1 = 0  # Ending index of the longest common substring in text1

        # Build the matrix in bottom-up manner
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if words1[i - 1].lower() == words2[j - 1].lower():
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > longest:
                        longest = dp[i][j]
                        end1 = i - 1  # Update the end index of the longest common substring
                else:
                    dp[i][j] = 0

        if longest == 0:
            return ((-1, -1), (-1, -1), "")  # No common substring found

        # Extract the longest common substring using the end index and length
        start1 = end1 - longest + 1
        end2 = dp[end1 + 1].index(longest) - 1
        start2 = end2 - longest + 1

        # Convert word indices back to the text substring
        common_substring = ' '.join(words1[start1:end1 + 1])

        return ((start1, end1), (start2, end2), common_substring)
    
    def calculate_iou_from_spans(span1, span2, total_words1, total_words2):
        """
        Calculate the Intersection over Union (IoU) from the spans of the longest common substrings.

        Args:
        span1 (tuple): Start and end indices of the span in text1.
        span2 (tuple): Start and end indices of the span in text2.
        total_words1 (int): Total number of words in text1.
        total_words2 (int): Total number of words in text2.

        Returns:
        float: The IoU score.
        """
        # Intersection is the length of the common substring
        intersection_length = span1[1] - span1[0] + 1

        # Calculate union
        union_length = total_words1 + total_words2 - intersection_length

        # Calculate IoU
        iou = intersection_length / union_length if union_length > 0 else 0
        return iou

    def score(self) -> None:
        text1 = self.setAnswer
        text2 = self.genAnswer

        words1 = word_tokenize(text1)
        words2 = word_tokenize(text2)

        self.result = self.find_longest_common_substring(text1, text2)
        self.iouScore = self.calculate_iou_from_spans(self.result[0], self.result[1], len(words1), len(words2))


    def getPrecision(self):
        pass
    
    #gets recall value for a given score
    def getRecall(self):
        pass

    #gets fmeasure value for a given score 
    def getfMeasure(self):
        pass

    def getAll(self) -> list:
        return self.iouScore
        
    def getName(self) -> str:
        return "IoUScore"