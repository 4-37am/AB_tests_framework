# Databricks notebook source
pip install pandas==1.5.3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pkg_resources
pkg_resources.require("pandas==1.5.3")

# COMMAND ----------

# MAGIC %run /python/helpers

# COMMAND ----------

# MAGIC %run AB-tests/ab_tests_related_functions_v2

# COMMAND ----------

# MAGIC %md
# MAGIC Input configurations for the sample size calculations

# COMMAND ----------

# MAGIC %md
# MAGIC Change a cell below if you estmate SS for **scenario with overall PACR or PISR** as acceptance metric. No other changes in the notebook needed in this case

# COMMAND ----------

prewritten_scenario = False # True/False
prewritten_scenario_params = {
  'metric': 'PACR' #only PACR and PISR are supported
  , 'MDE': 0.001 #change in PACR on the market you want to detect
  , 'sales_channels': ['Lounge AT'] #sales_channels planned for experiment
  , 'treatment_percent': 0.5 # choose min(share of treatment|control group) from all customers. E.g if you roll-out treatment group on 10% of customers in DE then filter your query with 'DE' filter and choose treatment_share_percent = 0.1  
  , 'additional_filters': "" # your filters. Example: "and device_type = 'Mobile' and payment_method_attempted = 'paypal'"
}

# COMMAND ----------

# MAGIC %md
# MAGIC Change cells below only if you estmate SS for a **scenario different** from the supported above

# COMMAND ----------

# NOTE that you can choose only 1 metric type and 1 metric within it to calculate needed Sample Size

# if your success metric is continuous one choose flag = TRUE and write name of the column for this metric from your query
continuous_metric_flag = True
continuous_metric_is_median = False
continuous_metric_name = 'gmv' 

# if your success metric is proportion metrc one choose flag = TRUE and write name of the column for these metrics in numerator:denumerator format from your query
proportions_metric_flag = False
proportions_columns = {'over100_customers':'all_customers'} # names of metrics from sql query that you want to use for proportion metrics in numerator:denumerator format. Note that values per user must be binary 0 or 1. 

# if your success metric is ratio metrc one choose flag = TRUE and write name of the column for these metrics in numerator:denumerator format from your query
ratio_metric_flag = False 
ratio_columns = {'items':'orders'} # names of metrics from sql query that you want to use for ratio metrics in numerator:denumerator format. 

max_weeks = 13 # choose max number of weeeks that you are ready to run your AB test
treatment_share_percent = 0.1 # choose share of treatment group from all customers. E.g if you roll-out treatment group on 10% of customers in DE then filter your query with 'DE' filter and choose treatment_share_percent = 0.1  
mde=0.01 # choose MDE that you want to detect in your AB test

min_weeks_with_significance = 3 # Choose number of weeks in-a-row with pvalue<0.05. The more you choose the more conservative and reliable results you will get but more weeks to wait for AB test results you will need. We suggest to use not less than 3. 5 is already very reliable

bootstrap_iterations=800 # (optional) if you want faster calculations you can reduce this number but you will lose in precision. 800 is recomnded as balance between precision and speed

# COMMAND ----------

if prewritten_scenario:
  continuous_metric_flag = False

  proportions_metric_flag = False

  ratio_metric_flag = True 
  ratio_columns = {'cuped_num_orders_placed':'cuped_num_rows'} 

  max_weeks = 30
  treatment_share_percent = prewritten_scenario_params['treatment_percent']
  mde = prewritten_scenario_params['MDE']
  min_weeks_with_significance = 3 

# COMMAND ----------

sample_share_percent=str(treatment_share_percent)
w=0
fast_mode = True
iterations = bootstrap_iterations
pval_counter=0
pvals=dict()
if (continuous_metric_flag and proportions_metric_flag) or (continuous_metric_flag and ratio_metric_flag) or (proportions_metric_flag and ratio_metric_flag):
  raise ValueError('You should choose  only one metric type flag')
  
while w<max_weeks and pval_counter<min_weeks_with_significance:
  w += 3 if fast_mode else 1
  wk = str(w)

  if w in pvals: 
    if pvals[w] < 0.05:
      pval_counter += 1
    else:
      pval_counter = 0
    print ('cashed: week=',w,'; pval=', pvals[w])
    continue
  
  #   insert your query that results into table grouped by customer_id and contains column with needed success metric calculations. Also you have to inject WHERE clause that contains 'date between dateadd('w',-"""+wk+""",current_date-1) and current_date-1'
  # If you want to calculate CUPED metrics you need also to collect pre experimental data with the same requirements as in the main AB test template https://docs.google.com/document/d/1jh4f5k_n13ioJTQFUKj1eX0eEpKiKVXnacj35fmm1oI/edit#heading=h.ryd4nyboinzm
  # You can use query after the row "if new_payment_method_scenario:" as a reference example
  query= f"""

  select
     sk_customer,
      sum(gmv_bef_return) as gmv,
      sum(total_items) as items,
      count(distinct order_number) as orders,
  case when gmv>100 then 1 else 0 end as over100_customers, 1 as all_customers
  from o, ch
  where o.sk_sales_channel=ch.sk_sales_channel
    and sk_payment_method = 3 
    and cancelation_status != 'FULL'
    and order_date::date between dateadd('w',-{wk},current_date-1) and current_date-1
  group by 1

  """
  if prewritten_scenario:
    if prewritten_scenario_params['metric'] == 'PACR':
      source_table = 'reporting.pam_purchase_attempt_funnel'
      datetime_column = 'purchase_attempt_created_at'
    if prewritten_scenario_params['metric'] == 'PISR':
      source_table = 'reporting.pam_order_placement_attempt_funnel'
      datetime_column = 'payment_initiated_at'
    query= f"""
    with metrics as (
      select
        customer_id
        , count(case 
            when {datetime_column} between dateadd('w',-{wk},current_date-1) and current_date 
            then 1 end
        ) as num_rows
        , count(case 
            when {datetime_column} between dateadd('w',-{wk},current_date-1) and current_date 
            then order_placed_at end
        ) as num_orders_placed
        , count(case 
            when {datetime_column} between dateadd('w',-{wk},current_date-365) and dateadd('w',-{wk},current_date-1) 
            then 1 end
        ) as pre_365_num_rows
        , count(case 
            when {datetime_column} between dateadd('w',-{wk},current_date-365) and dateadd('w',-{wk},current_date-1) 
            then order_placed_at end
        ) as pre_365_num_orders_placed
      from {source_table}
      where sales_channel in ('"""+"','".join(prewritten_scenario_params['sales_channels'])+f"""')
        and {datetime_column} between dateadd('w',-{wk},current_date-365) and current_date
        {prewritten_scenario_params['additional_filters']}
      group by 1
    )
    select *
    from metrics
    where num_rows > 0
    """
  
  pval, w, pval_counter, s_size, pvals = main_function_for_sample_size_calculations(query, mde, w, treatment_share_percent, pvals, pval_counter, proportions_columns, ratio_columns, continuous_metric_name, continuous_metric_flag, proportions_metric_flag, ratio_metric_flag, continuous_metric_is_median, bootstrap_iterations=iterations)  

  if pval < 0.05 and fast_mode:
    pval_counter = 0
    w -= 3
    fast_mode = False
  
  if pval < 0.1 and iterations == bootstrap_iterations:
    iterations = bootstrap_iterations*4 


pvals
plt.figure(figsize=(9, 7))
plt.axhline(y = 0.05, color = 'r', linestyle = '--')
plt.xlabel('weeks')
plt.ylabel('p-value')
x = sorted(pvals.keys())
y = [pvals[k] for k in x]
plt.plot(x, y,'o', ls='-')
if pval_counter>=min_weeks_with_significance:
  print ('To reach MDE=',mde,'with control group as',treatment_share_percent*100,'% from all audience, you will need ',w, 'weeks and ~',s_size,'customers')
else:
  print (max_weeks,' weeks is not enough to get MDE=',mde,' with control group as',treatment_share_percent*100,'% from all audience and ~',s_size,'customers')
  

# COMMAND ----------


