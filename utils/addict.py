from addict import Addict


def unchecked_merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    Ignores type mismatches, missing keys
    """

    merged_dict = b.deepcopy()

    if type(a) is not Addict:
        raise Exception(f"Type '{type(a)}' is not EasyDict")

    for k, v in a.items():
        # recursively merge dicts
        if type(v) is Addict and k in merged_dict:
            try:
                merged_dict[k] = unchecked_merge_a_into_b(a[k], merged_dict[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            merged_dict[k] = v

    return merged_dict
