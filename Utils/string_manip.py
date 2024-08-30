# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:40:24 2023

@author: 253863J
"""

def find_str_between(s, str1, str2):
    """
    Return string in input s between str1 and str2.

    Parameters
    ----------
    s : str
        Input string from which the return string is extracted.
    str1 : str
        String before returned string.
    str2 : str
        String after returned string.

    Returns
    -------
    str
        String between str1 and str2 in s.

    """
    try:
        i_start = s.index(str1) + len(str1)
        i_end = s.index(str2)
        return s[i_start:i_end]
    except ValueError:
        return ''    
    
def find_str_after_prefix(s, prefix, len_str):
    """
    Return string of len_str in input s placed after prefix.

    Parameters
    ----------
    s : str
        Input string from which the return string is extracted.
    prefix : str
        String before returned string.
    len_str : str
        Length of the extracted str.

    Returns
    -------
    str
        String of lengeth len_str after prefix.

    """
    try: 
        i_start = s.index(prefix) + len(prefix)
        return s[i_start:i_start+len_str]
    except ValueError:
        return ''   