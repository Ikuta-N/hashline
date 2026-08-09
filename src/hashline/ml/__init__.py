"""Semantic search.

``search`` is pure numpy and always importable. Anything that needs a model
lives behind the optional ``ml`` extra and is imported inside functions, so the
app runs fully without it.
"""
