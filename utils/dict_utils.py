class DictUtils:
    def insert_sub_dict(_dict, sub_dict):
        _dict.update(sub_dict)

    def dict_of_dict_average(dict_of_dict):
        result = {}
        for k1, v1 in dict_of_dict.items():
            for k2, v2 in v1.items():
                if result[k2] == None:
                    result[k2] = 0
                result[k2] = result[k2] + v2
        n = len(dict_of_dict)
        for k, v in result.items():
            result[k] /= n

    def dict_divide(t, n):
        return {k: v / n for k, v in t.iteritems()}

