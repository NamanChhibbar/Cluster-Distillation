'''
Contains text processing utilities for cleaning and normalizing text data.
'''
import re
import unicodedata

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from wtpsplit import SaT


class TextPreprocessor:

  _base_subs = [
    # Remove hyperlinks
    (r'https?://[^\s]+', ''),
    # Remove unecessary periods
    (r'\.\s*([,;:?!-])', r'\1'),
    (r'([,;:?!-])\s*\.', r'\1'),
    # Replace unicode quotes with standard quotes
    (r'[“”]', '"'),
    (r'[‘’]', "'"),
    # Replace unicode dashes with standard dash
    (r'[–—−‐]', '-'),
    # Replace unicode ellipsis with standard ellipsis
    (r'…', '...'),
    # Fix spaces before and after punctuations
    (r'\s+([,.;:?!])', r'\1'),
    (r'(?<=,)([^\s\d])', r' \1'),
    (r'(?<=[;:?!])(\S)', r' \1'),
    # Remove spaces within brackets and quotes
    (r"'([^']*)'", lambda m: f"'{m.group(1).strip()}'"),
    (r'"([^"]*)"', lambda m: f'"{m.group(1).strip()}"'),
    (r'\[([^\]]*)\]', lambda m: f'[{m.group(1).strip()}]'),
    (r'\(([^\)]*)\)', lambda m: f'({m.group(1).strip()})'),
    # Normalize non-newline spaces
    (r'[^\S\n]+', ' '),
    # Remove spaces around newline
    (r' *\n *', '\n'),
    # Remove lone newlines
    (r'(?<!\n)\n(?!\n)', ' '),
    # Replace multiple newlines
    (r'\n{3,}', '\n\n')
  ]

  # Matches everything except words, numbers, and single quotes
  _non_alphanumeric_subs = (r'[^\u0000-\u007f]', '')

  # Matches numbers with optional leading '+' or '-' sign
  _number_sub = (r'(\b|\+)[\d-]+\b', '')

  def __init__(
    self,
    only_alphanumeric: bool = False,
    no_nums: bool = False,
    ignore_tokens: list[str] | None = None
  ) -> None:
    '''
    Initializes the text preprocessor with specific patterns.
    Parameters:
      only_alphanumeric (bool): If True, removes non-alphanumeric characters.
      no_nums (bool): If True, removes numbers.
      ignore_tokens (list[str] | None): List of tokens to ignore during processing.
    '''
    pats_subs = TextPreprocessor._base_subs
    # Only words and numbers
    if only_alphanumeric:
      pats_subs.append(TextPreprocessor._non_alphanumeric_subs)
    # Remove numbers
    if no_nums:
      pats_subs.append(TextPreprocessor._number_sub)
    # Ignore specific tokens
    if ignore_tokens is not None:
      pats_subs.append((re.compile(r'|'.join(ignore_tokens)), ''))
    # Compile patterns
    self._pats_subs = [
      (re.compile(pat), sub) for pat, sub in pats_subs
    ]

  def __call__(self, texts: str | list[str]) -> str | list[str]:
    '''
    Processes a single text or list of texts.
    '''
    # Check if single text is given
    single_text = isinstance(texts, str)
    texts = [texts] if single_text else texts
    # Process texts
    processed_texts = []
    for text in texts:
      # Normalize unicode characters
      text = unicodedata.normalize('NFKD', text)
      text = ''.join(c for c in text if not unicodedata.combining(c))
      # Apply all patterns and substitutions
      for pat, sub in self._pats_subs:
        text = pat.sub(sub, text)
      # Strip leading and trailing spaces
      text = text.strip()
      processed_texts.append(text)
    return processed_texts[0] if single_text else processed_texts


class SegmenterEmbedder:
  '''
  A class to segment and embed text using SaT for segmentation and SentenceTransformer for embeddings.
  '''

  def __init__(self, device: str | torch.device = 'cpu') -> None:
    '''
    Initializes the SegmentEmbed class with a segmentation model and an embedding model.
    Args:
      device (str | torch.device): The device to run the models on (e.g., 'cuda', 'cpu').
    '''
    self.device = device
    self.seg_model = SaT('sat-6l-sm')
    self.seg_model.half().to(device)
    self.embed_model = SentenceTransformer('all-mpnet-base-v2', device=device)

  def __call__(self, text: str) -> tuple[list[str], np.ndarray]:
    # Segment the text
    segments: list[str] = self.seg_model.split(text)
    # Clean up segments
    segments = [seg.strip() for seg in segments if seg.strip()]
    # Embed the segments
    embeddings = self.embed_model.encode(segments, convert_to_numpy=True)
    return segments, embeddings


class BiMapping:
  '''
  A bidirectional mapping class that maps texts to embeddings.
  '''

  def __init__(self, texts: list[str], embeddings: np.ndarray) -> None:
    '''
    Initializes the BiMapping class with texts and their corresponding embeddings.
    Args:
      texts (list[str]): List of texts to be mapped.
      embeddings (torch.Tensor): Corresponding embeddings for the texts.
    '''
    # Use text as key
    self.text_to_embedding = dict(zip(texts, embeddings))
    # Use tuple of embeddings as key
    self.embedding_to_text = dict(
      (tuple(emb.tolist()), text)
      for emb, text in zip(embeddings, texts)
    )

  def __getitem__(self, key: str | np.ndarray) -> np.ndarray | str:
    '''
    Get embedding for text or vice-versa

    Parameters:
      key (str | np.ndarray): The key to retrieve the corresponding value.

    Returns:
      value (np.ndarray | str): The corresponding embedding if key is text, or the text if key is an embedding.
    '''
    if isinstance(key, str):
      return self.text_to_embedding[key]
    elif isinstance(key, np.ndarray):
      return self.embedding_to_text[tuple(key.tolist())]
    else:
      raise TypeError('Key must be either a string or a numpy array.')
