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

# MAGIC %run /AB-tests/ab_tests_related_functions_v2

# COMMAND ----------

# MAGIC %md
# MAGIC Input configurations for the experiment calculations

# COMMAND ----------

# inputs:

experiment_name = 'Test_of_new_feature_dictionary' # Name of experiment. Try to make it unique
test_start_date = '2022-06-10' # First date of roll-out of the experiment
calc_start_date = '2022-06-16' # First date of results calculation. Results well be calculate from test_start_date up to date between calc_start_date and calc_end_date.
calc_end_date = '2022-06-16' #If you need to calculate results just for 1 day make calc_start_date=calc_end_date
# yesterday=(pd.to_datetime("today") - pd.Timedelta(1, unit='D')).strftime('%Y-%m-%d')
calc_frequency = '7d' # frequency of calculations between calc_start_date and calc_end_date. Example: if calc_frequency='7d' and calc_start_date ='2022-01-01' and calc_end_date='2022-01-22' then calculations will be performed on: 2022-01-01, 2022-01-08,2022-01-15,2022-01-22. More info on frequency types: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases 
dates=[] # you can input list of dates to calculate result for each iteratively e.g. for [test_start_date;date_1],[test_start_date,date_2].. if yo want to calculate results daily or by given frequency just leave it empty. Note that number of dates in dates list should be > 2

metrics_columns = ['gmvbc','num_purchase_attempts','num_orders',{'gmvbc':'median'}
                   ,'cuped_gmvbc','cuped_num_purchase_attempts','cuped_num_orders'] # Names of metrics you need to calculate stat.tests. Should be the same as names in sql query results table.
proportions_columns = {'num_customers_with_morethan2_orders':'all_customers'} # names of metrics from sql query that you want to use for proportion metrics in numerator:denumerator format. Note that values per user must be binary 0 or 1. 
ratio_columns = {'num_orders':'num_purchase_attempts'} # names of metrics from sql query that you want to use for ratio metrics in numerator:denumerator format. 
dimensions_columns = [] #list of columns with the dimensions if you need any

need_bootstrap = True # True if you want to calculate Bootstrap statistics 
need_boostrap_wo_outliers = True # True if you want to calculate Bootstrap statistics based on data with removed outliers
zps_kpis_mode = WITH_KPI
# WITH_KPI - if you want to get ZPS KPIs (GMV,PACR,PSSR,PPSR) calculated by default
# KPI_WITH_CUPED - if you want to get same metrics as in WITH_KPI mode + CUPED versions of these metrics
# NO_KPI - no zps_kpi metrics
# ONLY_CUPED_KPI - if you want to get only CUPED versions of these metrics

experiment_id = " 'f1e6574c-a5a5-4f51-be73-c472e3bc91b9','b9b3b401-da13-434f-b970-224ce3b9c01e' " # could be one exp_id in "'exp_id'" format or several in "'exp_id1','exp_id2'" format
customer_id_column = 'customer_id' #name of column in sql resulted table that contains customer IDs
variant_col = 'payload_variant' #name of column in sql resulted table that contains names of treatment and control groups
control_gr_name = 'Invoice' #name of control group in variant_col
treatment_gr_name = 'BNPL2.0' #name of treatment group in variant_col
gr_ratio = 50.0/50.0 # control group % / treatment group %

frequency_sales_channel = None # parameter for frequency split, available options: 
# None - no frequency split is needed
# "all" - all sales channels included in the frequency calculation
# "shop" - all "Shop%" sales channels are included in the frequency calculation
# "lounge" - all "Lounge%" sales channels are included in the frequency calculation
# "'<name as in the purchase_attempt_funnel>', '<name as in the purchase_attempt_funnel>', ... '<name as in the purchase_attempt_funnel>'", for example, "'Shop DE', 'Shop IT', 'Lounge BE'" - custom list of sales_channels included in the frequency calculation


precalculated_metrics_dictionary = {
# if you do not want to calculate the checkout_funnel_KPIs just delete the entry from the dictionary or comment it
   "checkout_funnel_kpis": {
   'channel': 'shop'
# 'shop' - calculates the main checkout metrics for shop only 
# 'lounge' - calculates the main checkout metrics for lounge only
# 'all' - calculates the main checkout metrics for all sales channels
   , 'pisr_payment_methods': ['credit_card'] # list payment methord for pisr if you want to see it by payment method
# for example ['invoice', 'credit_card']
   }
  , "checkout_friction_metrics": [CHECKOUT_FRICTION]
#  available options:
# CHECKOUT_FRICTION - checkout friction metrics calculated by all purchase attempts
# CHECKOUT_FRICTION_SUCCESSFUL_PA_ONLY - checkout friction metrics calculated only by successful purchase attempts
}

# COMMAND ----------

##// Input name of table where rusults should be placed (table should be created before the start of calculations. if not - uncomment and run cmd 7)
TargetTable = "zps_analytics.ab_test_calculations_new_approach"

# COMMAND ----------

# MAGIC %md
# MAGIC Insert SQL query in cmd9 that counts needed metrics grouped by customer_id and variant type. 
# MAGIC Use date parameters from config ('test_start_date' and 'day' as date between calc_start_date and calc_end_date):

# COMMAND ----------

df_results=pd.DataFrame()
  # create a list of dates for calculations
if len(dates)<3:
  dates=pd.date_range(calc_start_date,calc_end_date,freq=calc_frequency).strftime('%Y-%m-%d').tolist()
calculations_time=pd.to_datetime("today").strftime('%Y-%m-%d-%H-%M-%S')
print(dates)
for day in dates:

#  Actions needed!
#   insert your query with data grouped by customer_id and group type (control/treatment). Add date parameters for test_start_date and day of calculations.
    
    query= f"""
    with """+get_octopus_assignment(experiment_id, test_start_date, day)+"""

    , pre_metrics as (
      select ta.customer_id
          , coalesce(sum(gmv_bef_cancellation), 0)::float as pre_365_gmvbc

      from  cm
      inner join octopus_assignment as ta using(customer_id)
      where purchase_attempt_created_at between date_add('day',-365,'"""+test_start_date+"""') and '"""+test_start_date+"""' 
      group by 1
    )

    , metrics as (
     select ta.customer_id
         , ta.payload_variant
         , coalesce(sum(gmv_bef_cancellation), 0)::float as gmvbc
    from  cm
    inner join octopus_assignment as ta on cm.customer_id=ta.customer_id
    where purchase_attempt_created_at between '"""+test_start_date+"""' and date_add('day',1,'"""+day+"""')
    group by 1,2
    )
    
    select 
      m.*
      , p.pre_365_gmvbc
      , p.pre_365_num_purchase_attempts
      , p.pre_365_num_orders
    from metrics m
    left join pre_metrics p using(customer_id)
     

"""
    query_flat = None  #you can write here additional query, for more information https://docs.google.com/document/d/1jh4f5k_n13ioJTQFUKj1eX0eEpKiKVXnacj35fmm1oI/edit#heading=h.z71pmqeqe5dw
    df_day_results = main_ab_test_calculation_function(
      customer_id_column=customer_id_column
      , experiment_id=experiment_id
      , test_start_date=test_start_date
      , day=day
      , calc_start_date=calc_start_date
      , calc_end_date=calc_end_date
      , metrics_columns=metrics_columns
      , need_bootstrap=need_bootstrap
      , proportions_columns=proportions_columns
      , ratio_columns=ratio_columns
      , dimensions_columns=dimensions_columns
      , query=query
      , query_flat=query_flat
      , gr_ratio=gr_ratio
      , frequency_sales_channel=frequency_sales_channel
      , zps_kpis_mode=zps_kpis_mode
      , precalculated_metrics_dictionary=precalculated_metrics_dictionary
      )
    
    df_results = df_results.append(df_day_results, ignore_index=False)

df_results = df_results.sort_values(by=['metric'])
notebookId=dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("notebookId").get()
notebookUrl='https://zalando-e2.cloud.databricks.com/#notebook/'+notebookId
df_results['notebook_url']=notebookUrl
df_to_redshift = spark.createDataFrame(df_results)
df_to_redshift.createOrReplaceTempView("tmp")
df_results.head()


# COMMAND ----------

##// change nothing here, just run it
df_to_redshift_final = sql("""
select *
from tmp """)
display(df_to_redshift_final)

# COMMAND ----------

get_summary(df_results)

# COMMAND ----------

get_no_full_data_continuous_metrics(df_results, experiment_name)

# COMMAND ----------

#// if you need to remove rows corresponding to your experiment_name from results table:
sqlDelete = f"""DELETE FROM {TargetTable} WHERE experiment_name like '{experiment_name}%' and dt = '2022-06-16' """
dwh_send(sqlDelete)

# COMMAND ----------

##//  write tests results in Redshift TargetTable defined in cmd 6 
sqlPreRun = f"""select * from {TargetTable}"""
dwh_write(df_to_redshift_final, TargetTable, sqlPreRun)

# COMMAND ----------


