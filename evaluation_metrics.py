import numpy as np


def R_squared_OSXS(true_return, pred_return):
    '''
    true_return: true return
    pred_return: predicted return
    '''
    return (
        1 - (
            sum(np.power(
                (true_return - true_return.mean())
                - (pred_return - pred_return.mean()), 2))
            / sum(
                np.power(true_return - true_return.mean(), 2)
            )
        )
    )


def CW_test(true_pred_return):
    '''
    CW test answers this question: 
        is my model doing better than naive benchmark? 

    true_pred_return should be a data frame
    with columns = ['optionid', 'time', 'true_return', 'pred_return']

    example:
    true_pred_return = pd.DataFrame(
    {
        "optionid": X_test.optionid,
        "time": X_test.date_x,
        "true_return": y_test,
        "pred_return": y_pred
    }
    )
    '''
    c = 1 / true_pred_return.shape[0] * (
        true_pred_return.groupby(["optionid", "time"]).apply(
            lambda group: sum(
                np.power(group["true_return"], 2)
                - np.power(group["true_return"] - group["pred_return"], 2)
            )))
    return np.mean(c) / np.std(c)


def DM_test(true_pred_1vs2_return):
    '''
    DM test answers this question:
        is model 1 doing better than model 2?
    perform DM test, postive result means model 2 outperforms model 1 

    construction example:
    true_pred_1vs2_return = pd.DataFrame(
    {
        "optionid": X_test.optionid,
        "time": X_test.date_x,
        "true_return": y_test,
        "pred_return1": regressor.predict(X_test[used_characteristics]),
        "pred_return2": est.predict(X_test[used_characteristics])
    }
    )
    '''
    d = 1 / true_pred_1vs2_return.shape[0] * (
        true_pred_1vs2_return.groupby(["optionid", "time"]).apply(
            lambda group:
            sum(
                np.power(group["true_return"] - group["pred_return1"], 2)
                - np.power(group["true_return"] - group["pred_return2"], 2)
            )))
    return np.mean(d) / np.std(d) 
