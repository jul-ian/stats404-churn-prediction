# test helper_functions.pyimport numpy as npimport pandas as pdimport pytestfrom tests.helper_functions import get_lists_of_dtypes, yn_recodedef test_get_lists_of_dtypes():    test_df = pd.DataFrame({"INT":[1, 2, 3], "FLOAT":[1.0, 2.0, 3.0], "STR":['a', 'b', 'c']})    lst_str, lst_int, lst_flo = get_lists_of_dtypes(test_df)    assert ("INT" in lst_int) and ("FLOAT" in lst_flo) and ("STR" in lst_str)def test_yn_recode():    test_df = pd.DataFrame(['Yes', 'No', 'YES'], columns = ['yes_no'])    test_df['yes_no_recode'] = test_df['yes_no'].apply(yn_recode)    assert all(test_df["yes_no_recode"] == pd.Series([1, 0, 1]))