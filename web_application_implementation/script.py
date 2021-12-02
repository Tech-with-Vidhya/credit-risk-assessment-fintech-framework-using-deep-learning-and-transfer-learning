# Function to pre-process the web application input data, and perform model predictions and forecasts

def tm1_preprocess_data_and_predict(form_input_data_list):

    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import load_model

    # Loading the Pre-Trained Target Model 1 (TM1) for Regression Prediction of "credit_score"
    target_model_1 = load_model("models/target_task_1_credit_score_regression_optimal_model.h5")
    
    # Loading the Pre-Trained Target Model 2 (TM2) for Binary Classification Forecast of "default_risk"
    target_model_2 = load_model("models/target_task_2_default_risk_classification_optimal_model.h5")
    
    all_input_data_list = ['1_pub_rec', '2_pub_rec',
                           '1_delinq_2yrs', '2_delinq_2yrs', '3_delinq_2yrs',
                           '1_num_tl_120dpd_2m', '2_num_tl_120dpd_2m',
                           '1_pub_rec_bankruptcies', '2_pub_rec_bankruptcies',
                           '1_num_tl_90g_dpd_24m', '2_num_tl_90g_dpd_24m',
                           '1_num_accts_ever_120_pd', '2_num_accts_ever_120_pd',
                           '1_acc_now_delinq', '2_acc_now_delinq',
                           '1_num_tl_30dpd', '2_num_tl_30dpd',
                           '1_total_rec_late_fee', '2_total_rec_late_fee',
                           '1_num_rev_tl_bal_gt_0', '2_num_rev_tl_bal_gt_0', '3_num_rev_tl_bal_gt_0',
                           '1_percent_bc_gt_75', '2_percent_bc_gt_75', '3_percent_bc_gt_75',
                           '1_revol_util', '2_revol_util', '3_revol_util', '4_revol_util',
                           '1_il_util', '2_il_util', '3_il_util',
                           '1_max_bal_bc', '2_max_bal_bc', '3_max_bal_bc', '4_max_bal_bc',
                           '1_mo_sin_old_rev_tl_op', '2_mo_sin_old_rev_tl_op', '3_mo_sin_old_rev_tl_op',
                           '1_months_since_earliest_cr_line', '2_months_since_earliest_cr_line', '3_months_since_earliest_cr_line',
                           '1_open_acc', '2_open_acc', '3_open_acc',
                           '1_num_sats', '2_num_sats', '3_num_sats',
                           '1_mort_acc', '2_mort_acc', '3_mort_acc',
                           '1_inq_last_6mths', '2_inq_last_6mths', '3_inq_last_6mths',
                           '1_open_il_12m', '2_open_il_12m', '3_open_il_12m',
                           '1_num_tl_op_past_12m', '2_num_tl_op_past_12m', '3_num_tl_op_past_12m', '4_num_tl_op_past_12m', 
                           '5_num_tl_op_past_12m',       
                           '1_annual_inc', '2_annual_inc', '3_annual_inc', '4_annual_inc', '5_annual_inc',
                           '1_dti', '2_dti',
                           '1_emp_length_int', '2_emp_length_int', '3_emp_length_int', '4_emp_length_int', '5_emp_length_int']
    
    input_data_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0]
    
    
    # Creating an Empty List to Append the Matching Index Positions of the Form Input Values
    match_index = []
    
    # Retrieving the Matching Index Positions of the Form Input Values
    for i in all_input_data_list:
        for j in form_input_data_list:
            if i == j:
                match_index.append(all_input_data_list.index(i))
    #print(match_index)
    
    # Preparing the Input Data List with Matching Form Input Values for further Model Predictions
    for index, k in enumerate(input_data_list):
        for m in match_index:
            if m == index:
                input_data_list[index] = 1
    #print(input_data_list)
    
    # Converting the Input Data List to a Numpy Array
    input_data_array = np.array([input_data_list])

    # Converting the Input Data Numpy Array to a Pandas DataFrame
    input_data_df = pd.DataFrame(input_data_array)
    
    # Target Model 1 - Predicting the Credit Score Value and Rounding-off the Score Value
    tm1_reg_result_float = np.round_(target_model_1.predict(input_data_df))

    # Converting the Flaoting Point Prediction Value to an Integer
    tm1_reg_result = tm1_reg_result_float.astype(int)

    # Printing the Predicted Results
    #print(tm1_reg_result[0][0])
    
    # Target Model 2 - Forecasting the Default Risk (Credit Default) of an Individual
    tm2_clas_result_float = target_model_2.predict(input_data_df)
    
    # Converting the Floating Point Forecast Value to a Percentage
    tm2_clas_result = np.round(tm2_clas_result_float * 100)
    
    # Returning the Predicted Credit Score Value and the Forecasted Default Risk Probability Value to the Front-end UI
    return tm1_reg_result[0][0], tm2_clas_result[0][0]
    
    
    
    
    


    