"""Helpers for turning caller-supplied files into multipart parts."""

from __future__ import annotations

import os
from typing import Union

from ._transport import FilePart

# What a caller may hand us for a single file:
#   * a path on disk (str)
#   * a (filename, content) pair
#   * a (filename, content, content_type) triple
FileInput = Union[str, tuple]

_DEFAULT_CONTENT_TYPE = "application/octet-stream"


def to_part(item: FileInput, field_name: str) -> FilePart:
    """Normalize one file input into a ``(field, filename, bytes, type)`` part."""
    if isinstance(item, str):
        with open(item, "rb") as fh:
            content = fh.read()
        return (field_name, os.path.basename(item), content, _DEFAULT_CONTENT_TYPE)

    if isinstance(item, tuple):
        if len(item) == 2:
            filename, content = item
            content_type = _DEFAULT_CONTENT_TYPE
        elif len(item) == 3:
            filename, content, content_type = item
        else:
            raise ValueError("file tuple must be (filename, content) or (filename, content, content_type)")
        if isinstance(content, str):
            content = content.encode("utf-8")
        return (field_name, filename, content, content_type)

    raise TypeError(f"unsupported file input: {type(item)!r}")


def to_parts(items: FileInput | list[FileInput], field_name: str) -> list[FilePart]:
    """Normalize one file or a list of files into parts under ``field_name``."""
    if isinstance(items, list):
        return [to_part(it, field_name) for it in items]
    return [to_part(items, field_name)]
