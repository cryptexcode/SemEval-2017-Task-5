# -*- coding: utf-8 -*-
from __future__ import division, print_function

from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ['NGramTfidfVectorizer']


class NGramTfidfVectorizer(TfidfVectorizer):
    """Convert a collection of  documents objects to a matrix of TF-IDF features.

      Refer to super class documentation for further information
    """

    def build_analyzer(self):
        """Overrides the super class method

        Parameter
        ----------
        self

        Returns
        ----------
        analyzer : function
            extract content from document object and then applies analyzer

        """
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.content))
