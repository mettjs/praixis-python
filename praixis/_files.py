"""Helpers for turning caller-supplied files into multipart parts."""

from __future__ import annotations

import os
from typing import Union

from ._transport import FilePart

# What a caller may hand us for a single file:
#   * a path on disk (str)
#   * a (filename, content) pair
#   * a (filename, content, content_type) triple
#
# The filename is the server's primary format signal (and the document's stored
# identity), so prefer a .pdf/.docx/.txt extension; the part's content type is
# the server's fallback for extension-less names.
FileInput = Union[str, tuple]

_DEFAULT_CONTENT_TYPE = "application/octet-stream"

# Server-supported formats, keyed by filename extension.
_EXTENSION_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}


def _content_type_for(filename: str) -> str:
    """Infer the part's content type from the filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    return _EXTENSION_CONTENT_TYPES.get(ext, _DEFAULT_CONTENT_TYPE)


def to_part(item: FileInput, field_name: str) -> FilePart:
    """Normalize one file input into a ``(field, filename, bytes, type)`` part."""
    if isinstance(item, str):
        with open(item, "rb") as fh:
            content = fh.read()
        filename = os.path.basename(item)
        return (field_name, filename, content, _content_type_for(filename))

    if isinstance(item, tuple):
        if len(item) == 2:
            filename, content = item
            content_type = _content_type_for(filename)
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
