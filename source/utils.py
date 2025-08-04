def str2bool(v):
    """
    Converts a string to a boolean value.

    Args:
        v (str): Input string expected to be "true" or "false" (case-insensitive).

    Returns:
        bool or None: True if input is "true", False if "false", None otherwise.
    """
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        return None