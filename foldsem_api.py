from foldsem_api_train import foldsem_api_train
from foldsem_api_test import foldsem_api_test
def foldsem_api(train_filter_table_path, val_filter_table_path, test_filter_table_path, rule_file_path, ratio, tail, user, password):
    # get the train, val and test filter table path and then use the api to generate the rules from the train set and test it on the val and test set.

    # Once I get the rules as a string, then I will have to write code to find the number of unique predicates, n_rules, n_size
   # Actually I should be able to get the above quantities through the API itself.
   # Also I should be able to get the predictions of the model in a list. But I would have to ensure that the list is not being shuffelled when I have to calculate fidelity w.r.t a neural model

   # I need an option in the API to just have input train data and output is the model json file

    n_rules, n_preds, size = foldsem_api_train(train_filter_table_path, rule_file_path, ratio, tail, user, password)
   # Call the foldsem api test with the test and validation set
    acc_train, acc_val, acc_test, y_train_f, y_val_f, y_test_f = foldsem_api_test(train_filter_table_path, val_filter_table_path, test_filter_table_path, user, password)

    return acc_train, acc_val, acc_test, y_train_f, y_val_f, y_test_f, n_rules, n_preds, size
