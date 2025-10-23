"""
Utility functions to assist in API responses.
"""


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """
    Flattens a nested dictionary with recursion. New keys will be the
    concatenation of the parent key and the current key, separated by `sep`.

    :param d: The dictionary to flatten.
    :param parent_key: The base key to use for the new keys. Defaults to an
        empty string.
    :param sep: The separator to use between the parent key and the current key.
        Defaults to "_".
    :return dict: A flattened dictionary.
    """

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
