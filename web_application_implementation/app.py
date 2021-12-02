# Credit Risk Assessment Dual Solution Framework - Real-World Web Application Implementation

from flask import Flask, render_template, request
import numpy as np
import joblib

import script as sc

app = Flask(__name__)

@app.route("/")
def form():
    return render_template("form.html")


@app.route("/results", methods= ['POST'])
def predict():
    if request.method == "POST":
        
        
        # Extracting the Form Input Data
        pub_rec = request.form['pub_rec']
        delinq_2yrs = request.form['delinq_2yrs']
        num_tl_120dpd_2m = request.form['num_tl_120dpd_2m']
        pub_rec_bankruptcies = request.form['pub_rec_bankruptcies']
        num_tl_90g_dpd_24m = request.form['num_tl_90g_dpd_24m']
        num_accts_ever_120_pd = request.form['num_accts_ever_120_pd']
        acc_now_delinq = request.form['acc_now_delinq']
        num_tl_30dpd = request.form['num_tl_30dpd']
        total_rec_late_fee = request.form['total_rec_late_fee']
        num_rev_tl_bal_gt_0 = request.form['num_rev_tl_bal_gt_0']
        percent_bc_gt_75 = request.form['percent_bc_gt_75']
        revol_util = request.form['revol_util']
        il_util = request.form['il_util']
        max_bal_bc = request.form['max_bal_bc']
        mo_sin_old_rev_tl_op = request.form['mo_sin_old_rev_tl_op']
        months_since_earliest_cr_line = request.form['months_since_earliest_cr_line']
        open_acc = request.form['open_acc']
        num_sats = request.form['num_sats']
        mort_acc = request.form['mort_acc']
        inq_last_6mths = request.form['inq_last_6mths']
        open_il_12m = request.form['open_il_12m']
        num_tl_op_past_12m = request.form['num_tl_op_past_12m']
        annual_inc = request.form['annual_inc']
        dti = request.form['dti']
        emp_length_int = request.form['emp_length_int']
             
        
        form_input_data_list = request.form.to_dict()
        form_input_data_list = list(form_input_data_list.values())
        #input_data_list = list(map(int, input_data_list))
        
        # Calling the preprocess_data_and_predict() Function and pass the form data as input
        credit_score_prediction, credit_default_forecast = sc.tm1_preprocess_data_and_predict(form_input_data_list) 
            
        # Pass the Prediction to the HTML Template
        return render_template("results.html", prediction=credit_score_prediction, forecast=credit_default_forecast)
        #return render_template("form.html", prediction=credit_score_prediction, forecast=credit_default_forecast)
        
    pass
    
  

if __name__ == "__main__":
    app.run(debug=True)