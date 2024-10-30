# Databricks notebook source
from experimentation_platform_analysis.stats.preprocessing import winsorize
from experimentation_platform_analysis.stats.variance_reduction import apply_the_adjuster_and_readjustment, simple_OLS

# COMMAND ----------

pip install numba

# COMMAND ----------

import numpy as np
import pandas as pd
import numba as nb
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import time
import datetime
import matplotlib.pyplot as plt

from statsmodels.stats.proportion import confint_proportions_2indep

# COMMAND ----------

from functools import reduce
import pyspark.sql.functions as f

def coalesce_nulls_with_nonadj_value (df, metrics, suffix):
    return( reduce(lambda  df, idx: df.withColumn('adj_' + suffix  + metrics[idx],
        [f.coalesce(f.col('adj_' + suffix  + col), f.col(col)) for col in metrics][idx] ),range(len(metrics)), df) )

# COMMAND ----------

def get_octopus_assignment(experiment_id=None, start_date=None, end_date=None, assignment_key='customer_id'):
  if assignment_key in ( 'customer_id','customer_hash') :
    return """
octopus_assignment as (
select
  """ + ( "vc.customer_number " if  assignment_key == 'customer_hash' else "customer_number " ) + """ as customer_id
  , any_value(payload_variant) as payload_variant
  , min(occurred_at) as occurred_at
  , min(occurred_at)::date as assignment_on
from octopus_feedback
  """ + ( "left join dwh.v_customer vc on vc.customer_number_hash = payload_customer_hash " if  assignment_key == 'customer_hash' else "" ) + """
where
   payload_mode in ('CLIENT','TEST')
   """ + ("and payload_experiment_id in ( "+experiment_id+")" if experiment_id else "") + """
   """ + ("and occurred_at >= '"+start_date+" 00:00:00'" if start_date else "") + """
   """ + ("and occurred_at <= '"+end_date+" 23:59:59'" if end_date else "") + """
group by 1
having count(distinct payload_variant) = 1
       and sum(case when payload_reason in ('TECHNICAL_ERROR', 'TIMEOUT') then 1 else 0 end) = 0
)
"""
  else:
    raise Exception("Sorry, your assignment_key is not supported")
    

# COMMAND ----------

WITH_KPI = 1
KPI_WITH_CUPED = 2
NO_KPI = 3
ONLY_CUPED_KPI = 4

def zpi_kpis_lists(type='metric', mode=KPI_WITH_CUPED):
  prefix='cuped_'
  res = {}
  if type == 'metric':
    res_tmp = ['zps_kpi_gmvbc','zps_kpi_num_purchase_attempts','zps_kpi_num_orders_placed']
    res = []
    if mode in (WITH_KPI, KPI_WITH_CUPED):
      res += res_tmp
    if mode in (KPI_WITH_CUPED, ONLY_CUPED_KPI):
      res += [prefix+x for x in res_tmp]
  elif type == "ratio": 
    res_tmp = {
      'zps_kpi_num_purchase_attempts_reached_payment_selected_with_rendered': ['zps_kpi_num_purchase_attempts_with_payment_methods_rendered']
      , 'zps_kpi_num_orders_placed': ['zps_kpi_num_purchase_attempts_reached_payment_initiated', 'zps_kpi_num_purchase_attempts']
      , 'zps_kpi_payment_exits':['zps_kpi_num_purchase_attempts']
      , 'zps_kpi_gmvbc': ['zps_kpi_num_orders_placed']
    }
    res = {}
    if mode in (WITH_KPI, KPI_WITH_CUPED):
      for k, v in list(res_tmp.items()):
        res[k] = v
    if mode in (KPI_WITH_CUPED, ONLY_CUPED_KPI):
      for k, v in list(res_tmp.items()):
        res[prefix+k] = [prefix + m for m in v]
  elif type == 'proportion':
    res['zps_cuca_case_customers'] ='zps_all_pa_customers'
  return res


def date_add(d, days=0, date_format = "%Y-%m-%d"):
  return datetime.datetime.strftime(datetime.datetime.strptime(d, date_format) + datetime.timedelta(days=days), date_format)


def get_zps_kpis(start_date, end_date, experiment_id=None, assignment_key='customer_id', prefix=''):
  return f"""
    """+prefix+"""cuca as (
      select vc.customer_number as """+prefix+"""customer_id
        , order_number
        , count(case when order_number is not null or cc.created_date < date_add('day',1,'"""+end_date+"""') then case_number end) as c_cases
      from  cc
          , t
          ,  vc
      where cc.created_date >= '"""+start_date+"""'
        and t.high_level_contactcategory in ('Payment process', 'Refunds', 'Return process', 'Other reasons')
        and cc.sk_kunden = vc.sk_customer
        and cc.sk_case_category = t.sk_case_category
      group by 1, 2
    )
    """ + (","+get_octopus_assignment(experiment_id, start_date, end_date, assignment_key) if experiment_id else "") + """
    , """+prefix+"""order_level_join as (
      select paf.customer_id as """+prefix+"""customer_id
         , coalesce(sum(gmv_bef_cancellation), 0.0)::float as """+prefix+"""zps_kpi_gmvbc
         , count(case when  funnel_step_exited in ('2. Payment Selection Page','4. Order Confirmation Clicked Invalid','6. Payment Initiation') then purchase_attempt_id else null end) as  """+prefix+"""zps_kpi_payment_exits
    from  paf
    """ + ("inner join octopus_assignment oa on oa.customer_id = paf.customer_id" if experiment_id else "") + """
    left join cuca cc on cc.customer_id = paf.customer_id and cc.order_number is not null and cc.order_number = paf.order_number_placed
    where purchase_attempt_created_at between '"""+start_date+"""' and date_add('day',1,'"""+end_date+"""')
    """ + ("and assignment_on <= purchase_attempt_created_at" if experiment_id else "") + """
    group by 1
  )
  , """+prefix+"""zps_kpis as (
    select o.*
      ,  """+prefix+"""zps_cuca_cases_with_order + coalesce(cc.c_cases,0) as  """+prefix+"""zps_cuca_cases
      , case when  """+prefix+"""zps_cuca_cases > 0 then 1 else 0 end as  """+prefix+"""zps_cuca_case_customers
    from """+prefix+"""order_level_join o
    left join """+prefix+"""cuca cc on cc."""+prefix+"""customer_id = o."""+prefix+"""customer_id and cc.order_number is null
  )
  """

def get_zps_kpis_query(start_date, end_date, experiment_id=None, with_pre=False, assignment_key='customer_id'):
  pre_period = 365
  if not with_pre:
    return """
    with 
    """+ get_zps_kpis(start_date, end_date, experiment_id, assignment_key, prefix='' ) + """
    select * from zps_kpis
    """
  return"""
  with 
  """+ get_zps_kpis(start_date, end_date, experiment_id, assignment_key, prefix='') + """
  , 
  """+ get_zps_kpis(date_add(start_date, days=-pre_period), date_add(start_date, days=-1)
                    , experiment_id=None, assignment_key=None,  prefix='pre_'+str(pre_period)+ '_')+"""
  select *
  from zps_kpis m
  left join pre_"""+str(pre_period)+"""_zps_kpis h on m.customer_id = h.pre_"""+str(pre_period)+"""_customer_id
  """

# COMMAND ----------

#Dictionary related functions starting from this line

# COMMAND ----------

def checkout_funnel_lists(type='metric', configuration={}):
  res = {}
  if type == "ratio": 
    res = {
      'checkout_funnel_kpi_purchase_attempts_with_payment_methods_rendered': ['checkout_funnel_kpi_purchase_attempts']
    }
    for payment_method in configuration['pisr_payment_methods']:
      res['checkout_funnel_kpi_pisr_num_orders_placed_'+payment_method] = [
        'checkout_funnel_kpi_pisr_num_orders_payment_initiated_'+payment_method,

      ]

  return res


def get_checkout_funnel_query(start_date, end_date, configuration, experiment_id=None, assignment_key='customer_id'):
  if configuration is None:
        raise Exception("Invalid configuration for the checkout funnel related metrics: missing 'channel' or 'pisr_payment_methods'."
        )
  query = 'with'
  if experiment_id is not None:
    query += get_octopus_assignment(experiment_id, start_date, end_date, assignment_key) + ','
  query += """
  pisr as (
      select r.customer_id
       , count(distinct order_number_attempted) as num_orders_payment_initiated
       , count(distinct case when order_placed_at is not null then join_hash end) as num_orders_placed"""
  for payment_method in configuration['pisr_payment_methods']:
      query += """
        , count(distinct case when payment_method_attempted = '"""+payment_method+"""' then order_number_attempted end) as num_orders_payment_initiated_"""+payment_method+"""
        , count(distinct case when payment_method_attempted = '"""+payment_method+"""' and order_placed_at is not null then join_hash end) as num_orders_placed_"""+payment_method
  query += """
    from reporting.pam_order_placement_attempt_funnel r
  """
  if experiment_id is not None:
    query += 'inner join octopus_assignment o using(customer_id)'
  query += """
    where session_created_at between '"""+start_date+"""' and date_add('day',1,'"""+end_date+"""')
    """
  if experiment_id is not None:
    query += 'and assignment_on <= session_created_at'
  if configuration['channel'] != 'all':
    query += """
    and lower(split_part(sales_channel,' ',1)) = '"""+configuration['channel']+"""'
    """
  query += """
    group by 1
  )
  """
  query += """
    , checkout_funnel as (
    select paf.customer_id
       , count(purchase_attempt_id) as checkout_funnel_kpi_purchase_attempts
       , count(case when funnel_saw_payment_page then purchase_attempt_id end) as checkout_funnel_kpi_purchase_attempts_with_payment_methods_rendered
       , count(case when funnel_reached_payment_selection then purchase_attempt_id end) as checkout_funnel_kpi_purchase_attempts_reached_payment_selection
       , count(case when funnel_reached_payment_selected then purchase_attempt_id end) as checkout_funnel_kpi_purchase_attempts_reached_payment_selected
       , count(case when funnel_reached_order_placement_attempted then purchase_attempt_id end) as checkout_funnel_kpi_purchase_attempts_reached_order_placement_attempted
       , count(case when funnel_reached_payment_initiated then purchase_attempt_id end) as checkout_funnel_kpi_purchase_attempts_reached_payment_initiated
       , count(case when funnel_reached_order_placed then purchase_attempt_id end) as checkout_funnel_kpi_purchase_attempts_reached_order_placed
       , count(case when funnel_step_exited in ('2. Payment Selection Page','4. Order Confirmation Clicked Invalid','6. Payment Initiation') THEN purchase_attempt_id ELSE NULL END) as checkout_funnel_kpi_payment_exits
    from  paf
  """
  if experiment_id is not None:
    query += 'inner join octopus_assignment oa using(customer_id)'
  query += """
    where purchase_attempt_created_at between '"""+start_date+"""' and date_add('day',1,'"""+end_date+"""')
    """
  if configuration['channel'] != 'all':
    query += """
    and lower(split_part(sales_channel,' ',1)) = '"""+configuration['channel']+"""'
    """
  if experiment_id is not None:
    query += 'and assignment_on <= purchase_attempt_created_at'
  query += """
    group by 1
  )
  select
    cf.*
    , coalesce(pisr.num_orders_payment_initiated, 0) as checkout_funnel_kpi_pisr_num_orders_payment_initiated
    , coalesce(pisr.num_orders_placed, 0) as checkout_funnel_kpi_pisr_num_orders_placed
    """
  for payment_method in configuration['pisr_payment_methods']:
      query += """
    , coalesce(pisr.num_orders_payment_initiated_"""+payment_method+""", 0) as checkout_funnel_kpi_pisr_num_orders_payment_initiated_"""+payment_method+"""
    , coalesce(pisr.num_orders_placed_"""+payment_method+""", 0) as checkout_funnel_kpi_pisr_num_orders_placed_"""+payment_method
  query += """
  from checkout_funnel cf 
  left join pisr using(customer_id)
  """
  return query

# COMMAND ----------

CHECKOUT_FRICTION = 'checkout_friction'
CHECKOUT_FRICTION_SUCCESSFUL_PA_ONLY = 'checkout_friction_successful_pa_only'

def checkout_friction_lists(type='metric', configuration=[]): 
  res = {}
  if type == "ratio": 
    res = {#'friction_successful_pi_time_sum': ['friction_succesful_pa']
      # , 'friction_pi_time_sum': ['friction_pi_time_denum']
    }
    if CHECKOUT_FRICTION_SUCCESSFUL_PA_ONLY in configuration:
      res['friction_succesful_pa_with_friction'] = ['friction_succesful_pa']
      res['friction_succesful_pa_with_left_skip_flow'] = ['friction_succesful_pa']
      res['friction_succesful_pa_with_loop_on_payment_selection_page'] = ['friction_succesful_pa']
      res['friction_succesful_pa_with_unsuccessful_payment_initiation'] = ['friction_succesful_pa']
      res['friction_succesful_pa_with_unsuccessful_risk_check'] = ['friction_succesful_pa']

    if CHECKOUT_FRICTION in configuration:
      res['friction_pa_with_friction'] = ['friction_pa']
      res['friction_pa_with_left_skip_flow'] = ['friction_pa']
      res['friction_pa_with_loop_on_payment_selection_page'] = ['friction_pa']
      res['friction_pa_with_unsuccessful_payment_initiation'] = ['friction_pa']
      res['friction_pa_with_unsuccessful_risk_check'] = ['friction_pa']

  return res


def get_checkout_friction_query(start_date, end_date, configuration, experiment_id=None, assignment_key='customer_id'):
  query = 'with'
  if experiment_id is not None:
    query += get_octopus_assignment(experiment_id, start_date, end_date, assignment_key)
  query += """
 , funnel_enriched as (
    select
        flow_category = 'Left Skip Flow' as friction_left_skip_flow
        , payment_methods_rendered_at_min is not null and payment_methods_rendered_at_min != payment_methods_rendered_at_max as friction_loop_with_payment_selection_page
        , num_orders_payment_initiated > num_orders_placed as friction_unsuccessful_payment_initiation
        , num_order_placement_attempts > num_orders_payment_initiated as friction_unsuccessful_risk_check
        , *
    from reporting.pam_purchase_attempt_funnel paf
  """
  if experiment_id is not None:
    query += 'inner join octopus_assignment oa using(customer_id)'
  query += """
    where purchase_attempt_created_at between '"""+start_date+"""' and date_add('day',1,'"""+end_date+"""')
    """
  if experiment_id is not None:
    query += 'and assignment_on <= purchase_attempt_created_at'
  query += """
  )
  select customer_id
    , count(case when order_number_placed is not null then 1 end) as friction_succesful_pa
  """
  if  CHECKOUT_FRICTION_SUCCESSFUL_PA_ONLY in configuration:
    query += """
    , count(case when order_number_placed is not null and (friction_left_skip_flow or friction_loop_with_payment_selection_page or friction_unsuccessful_payment_initiation
                       or friction_unsuccessful_risk_check)
            then 1 end) as friction_succesful_pa_with_friction


  """
  if CHECKOUT_FRICTION in configuration:
    query += """  
    , count(*) as friction_pa
    , count(case when (friction_left_skip_flow or friction_loop_with_payment_selection_page or friction_unsuccessful_payment_initiation
                       or friction_unsuccessful_risk_check)

  """
  query += """
    /*
    , sum(case when order_number_placed is not null then date_diff('seconds',order_confirmation_clicked_at_max, order_placed_at) end) as friction_successful_pi_time_sum
    , sum(case when order_confirmation_clicked_at_max is not null then coalesce(date_diff('seconds',order_confirmation_clicked_at_max, order_placed_at), 900) end) as friction_pi_time_sum
    , count(case when order_confirmation_clicked_at_max is not null then 1 end) as friction_pi_time_denum
    */
from funnel_enriched
group by 1
  """
  return query


# COMMAND ----------

dictionary_with_functions= {
    'checkout_funnel_kpis': get_checkout_funnel_query,
    'checkout_friction_metrics': get_checkout_friction_query
    }

# COMMAND ----------

def get_customer_frequency_query(start_date, end_date, sales_channel='all'):
    return """
    select
        customer_id
        , count(*) as customer_frequency_purchase_attempts
        , case when customer_frequency_purchase_attempts > 1 then 'high'
            else 'low' end as purchase_attempt_frequency
        , case when customer_frequency_purchase_attempts > 1 then 1
            else 0 end as customers_with_high_frequency
    from reporting.pam_purchase_attempt_funnel
    where purchase_attempt_created_at between '"""+start_date+"""' and date_add('day',1,'"""+end_date+"""')
    """ +('' if sales_channel == 'all' else (
        "and sales_channel like 'Shop%'" if sales_channel == 'shop' else (
        "and sales_channel like 'Lounge%'" if sales_channel == 'lounge' else
        "and sales_channel in ("+sales_channel+")"
    ))) + """
    group by 1
    """

# COMMAND ----------


# functions needed for further calculations:
def relative_diff(num,denum):
  if denum==0:
    return np.NaN
  else:
    return (num-denum)/abs(denum)


def array_diff(tr,ctr):
  return tr-ctr

# removes outliers with given quantile
def remove_outliers(df_customer,metric,quantile_for_outliers):
  outliers_threshold=df_customer[metric].astype('float').quantile(quantile_for_outliers)
  df_wo_outliers = df_customer.drop(df_customer[(df_customer[metric] > outliers_threshold)].index)
  return df_wo_outliers,outliers_threshold

# return samples based on column and metrics names
def get_control_and_treatment_samples(df_customer,metric,variant_col,control_gr_name,treatment_gr_name):
  control=df_customer[metric][df_customer[variant_col] == control_gr_name].values
  control=control[~np.isnan(control)]
  treatment=df_customer[metric][df_customer[variant_col] == treatment_gr_name].values
  treatment=treatment[~np.isnan(treatment)]
  return control, treatment


def get_control_and_treatment_samples_num_denum(df_customer,metric_num, metric_denum,variant_col,control_gr_name,treatment_gr_name):
  res = {}
  for gr_name in (control_gr_name,treatment_gr_name):
    num = df_customer[metric_num][df_customer[variant_col] == gr_name].values
    denum = df_customer[metric_denum][df_customer[variant_col] == gr_name].values
    not_nan_indexes = (~np.isnan(num)) & (~np.isnan(denum))
    res[gr_name] = (num[not_nan_indexes],denum[not_nan_indexes])
  return res[control_gr_name][0], res[treatment_gr_name][0], res[control_gr_name][1], res[treatment_gr_name][1]

# return z-test CI  
def get_z_test_CI(control,treatment):
  ci_diff_primary = 1.96 * np.sqrt(np.square(np.std(treatment)) / len(treatment) + np.square(np.std(control)) / len(control))
  return ci_diff_primary

@nb.njit(parallel=True)
def means_parallel_shuffle(arr, iterations, sample_size):
    size = len(arr)
    res = np.empty(iterations, np.float64)
    for i in nb.prange(iterations):
        tmp = np.random.randint(0, size, sample_size)
        s = [arr[tmp[j]] for j in range(sample_size)]
        res[i] = sum(s) / sample_size
    return res

# bootstrap
def get_bootstrap_results(control,treatment,iterations=800,bootstrap_median=False):
  sample_size = min(len(control), len(treatment))
  if not bootstrap_median:
    boot_means_ctr = means_parallel_shuffle(control, iterations, sample_size)
    boot_means_tr = means_parallel_shuffle(treatment, iterations, sample_size)
  else:
    boot_means_ctr = np.median(np.random.choice(control, size=(iterations, sample_size), replace=True),axis=1)
    boot_means_tr = np.median(np.random.choice(treatment, size=(iterations, sample_size), replace=True),axis=1)
  boot_mean_diff = boot_means_tr - boot_means_ctr
  k = sum(1 for x in boot_mean_diff if x < 0)
  CI1_means = np.percentile(boot_mean_diff, [2.5, 97.5])

  pval = 2 * np.minimum(k, iterations - k) / iterations
  ctr_mean = np.mean(boot_means_ctr)
  tr_mean = np.mean(boot_means_tr)
  mean_diff = relative_diff(tr_mean,ctr_mean)
  ci_prc_low = CI1_means[0]/abs(ctr_mean) if ctr_mean != 0 else 0.0
  ci_prc_up = CI1_means[1]/abs(ctr_mean) if ctr_mean != 0 else 0.0

  return np.round(pval,4),np.round(ctr_mean,4),np.round(tr_mean,4),np.round(mean_diff,5),np.round(ci_prc_low,5),np.round(ci_prc_up,5)


# get buckets and apply t_test
def get_buckets_results(control,treatment,buckets=150,confidence_level = 0.95):
  buckets_means_ctr=[]
  buckets_means_tr=[]
  means_diff=[]
  control_available_buckets=control
  treatment_available_buckets=treatment
  sample_size=int(np.floor(len(control)/buckets))
  if sample_size<1000:
    buckets=int(np.floor(len(control)/1000))
    sample_size=int(np.floor(len(control)/buckets))
  
  for b in np.arange(buckets):
    if (sample_size<=len(control_available_buckets)) & (sample_size<=len(treatment_available_buckets)):
      control_idx=np.random.randint(0, len(control_available_buckets), sample_size)
      test_idx=np.random.randint(0, len(treatment_available_buckets), sample_size)
      sample_ctr=control_available_buckets[control_idx]
      sample_tr=treatment_available_buckets[test_idx]
      tr_mean=np.mean(sample_tr)
      ctr_mean=np.mean(sample_ctr)
      means_diff.append(tr_mean-ctr_mean)
      buckets_means_ctr.append(ctr_mean)
      buckets_means_tr.append(tr_mean)
      control_available_buckets = np.delete(control_available_buckets, control_idx)
      treatment_available_buckets= np.delete(treatment_available_buckets, test_idx)

  pval_mean=stats.ttest_ind(buckets_means_ctr,buckets_means_tr).pvalue
  mean_buckets_tr=np.mean(buckets_means_tr)
  mean_buckets_ctr=np.mean(buckets_means_ctr)
  
  degrees_freedom = len(means_diff) - 1
  sample_mean = np.mean(means_diff)
  sample_standard_error = stats.sem(means_diff)
  confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
  ci_low=confidence_interval[0]
  ci_up=confidence_interval[1]
  if mean_buckets_ctr!=0:
    ci_prc_low=ci_low/abs(mean_buckets_ctr)
    ci_prc_up=ci_up/abs(mean_buckets_ctr)
  else:
    ci_prc_low=0.0
    ci_prc_up=0.0
  
   
  return np.round(pval_mean,4),np.round(mean_buckets_ctr,4),np.round(mean_buckets_tr,4),np.round(relative_diff(mean_buckets_tr,mean_buckets_ctr), 5),np.round(ci_prc_low,5),np.round(ci_prc_up,5)


# return t-test pval,CIs, mean diffs
def get_t_test_results(ctr,tr,confidence_level = 0.95):
  ctr=ctr.astype('float')
  tr=tr.astype('float')
  pval=stats.ttest_ind(ctr,tr).pvalue
  mean_tr=np.mean(tr)
  mean_ctr=np.mean(ctr)
  if len(tr)>len(ctr):
    tr=np.random.choice(tr, size=len(ctr), replace=False)
  else:
    ctr=np.random.choice(ctr, size=len(tr), replace=False)
  diff=array_diff(tr,ctr)
  degrees_freedom = len(diff) - 1
  sample_mean = np.mean(diff)
  sample_standard_error = stats.sem(diff)
  confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
  ci_low=confidence_interval[0]
  ci_up=confidence_interval[1]
  if mean_ctr!=0:
    ci_prc_low=ci_low/abs(mean_ctr)
    ci_prc_up=ci_up/abs(mean_ctr)
  else:
    ci_prc_low=0.0
    ci_prc_up=0.0
  
  return np.round(pval,4),np.round(mean_tr,4),np.round(mean_ctr,4),np.round(relative_diff(mean_tr,mean_ctr),5),np.round(ci_prc_low,5),np.round(ci_prc_up,5)

# get results of proprtion test
def get_chi2_test_results(control_num,control_denum,treatment_num,treatment_denum):
  sum_control_num = np.sum(control_num)
  sum_control_denum = np.sum(control_denum)
  sum_treatment_num = np.sum(treatment_num)
  sum_treatment_denum = np.sum(treatment_denum)
  p_control = sum_control_num/sum_control_denum
  p_treatment = sum_treatment_num/sum_treatment_denum
  p_prc_diff = np.round(((p_treatment/p_control)-1),4)
  p_diff = p_treatment - p_control
  p_control = np.round(p_control, 4)
  p_treatment = np.round(p_treatment, 4)
  if sum_control_num == 0 or (sum_control_denum - sum_control_num == 0 and sum_treatment_denum - sum_treatment_num == 0):
    print('Observed frequencies are 0')
    return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,  p_control, p_treatment, p_diff, p_prc_diff
  T = np.array([[sum_control_num, sum_control_denum - sum_control_num], [sum_treatment_num, sum_treatment_denum - sum_treatment_num]])
  conv_pval=np.round(stats.chi2_contingency(T,correction=False)[1],5)
  ci_low,ci_up=confint_proportions_2indep(sum_treatment_num,sum_treatment_denum,sum_control_num,sum_control_denum,alpha=0.05,compare='diff')
  ci_prc_low=np.round(ci_low/p_control,4)
  ci_prc_up=np.round(ci_up/p_control,4)
 

  return conv_pval, ci_low, ci_up, ci_prc_low, ci_prc_up, p_control, p_treatment, p_diff, p_prc_diff

@nb.njit(parallel=True)
def means_parallel_shuffle_conversions(arr, iterations, sample_size):
    size = len(arr)
    res = np.empty(iterations, np.float64)
    for i in nb.prange(iterations):
        tmp = np.random.randint(0, size, sample_size)
        num = [arr[tmp[j]][0] for j in range(sample_size)]
        denum = [arr[tmp[j]][1] for j in range(sample_size)]
        res[i] = sum(num) / sum(denum)
    return res
  
def get_bootstrap_results_for_conversions(control_num_arr, control_denum_arr, tr_num_arr, tr_denum_arr, iterations=800):
  sample_size=min(len(control_denum_arr),len(tr_denum_arr))
  control = np.array([(control_num_arr[i], control_denum_arr[i]) for i in range(len(control_denum_arr))])
  treatment = np.array([(tr_num_arr[i], tr_denum_arr[i]) for i in range(len(tr_denum_arr))])
  boot_means_ctr = means_parallel_shuffle_conversions(control, iterations, sample_size)
  boot_means_tr = means_parallel_shuffle_conversions(treatment, iterations, sample_size)
  boot_mean_diff = boot_means_tr - boot_means_ctr
  k = sum(1 for x in boot_mean_diff if x < 0)
  CI1_means = np.percentile(boot_mean_diff, [2.5, 97.5])

  pval=2 * np.minimum(k, iterations - k) / iterations
  ctr_mean=np.mean(boot_means_ctr)
  tr_mean=np.mean(boot_means_tr)
  mean_diff=relative_diff(tr_mean,ctr_mean)
  ci_prc_low=CI1_means[0]/abs(ctr_mean)
  ci_prc_up=CI1_means[1]/abs(ctr_mean)
  return np.round(pval,4),np.round(ctr_mean,4),np.round(tr_mean,4),np.round(mean_diff,5),np.round(ci_prc_low,5),np.round(ci_prc_up,5)

  
def get_buckets_results_for_conversions(control_num_arr, control_denum_arr, tr_num_arr, tr_denum_arr,buckets=150,confidence_level = 0.95):
  buckets_means_ctr=[]
  buckets_means_tr=[]
  means_diff=[]
  control_num_available_buckets=control_num_arr
  treatment_num_available_buckets=tr_num_arr
  control_denum_available_buckets=control_denum_arr
  treatment_denum_available_buckets=tr_denum_arr
  sample_size=int(np.floor(len(control_num_arr)/buckets))
  if sample_size<4000:
    buckets=int(np.floor(len(control_num_arr)/4000))
    sample_size=int(np.floor(len(control_num_arr)/buckets))
  
  for b in np.arange(buckets):
    if (sample_size<=len(control_num_available_buckets)) & (sample_size<=len(treatment_num_available_buckets)):
      control_idx=np.random.randint(0, len(control_num_available_buckets), sample_size)
      test_idx=np.random.randint(0, len(treatment_num_available_buckets), sample_size)
      sample_ctr=control_num_available_buckets[control_idx]/control_denum_available_buckets[control_idx]
      sample_tr=treatment_num_available_buckets[test_idx]/treatment_denum_available_buckets[test_idx]
      tr_mean=np.mean(sample_tr)
      ctr_mean=np.mean(sample_ctr)
      means_diff.append(tr_mean-ctr_mean)
      buckets_means_ctr.append(ctr_mean)
      buckets_means_tr.append(tr_mean)
      control_num_available_buckets = np.delete(control_num_available_buckets, control_idx)
      treatment_num_available_buckets= np.delete(treatment_num_available_buckets, test_idx)
      control_denum_available_buckets = np.delete(control_denum_available_buckets, control_idx)
      treatment_denum_available_buckets= np.delete(treatment_denum_available_buckets, test_idx)

  pval_mean=stats.ttest_ind(buckets_means_ctr,buckets_means_tr).pvalue
  mean_buckets_tr=np.mean(buckets_means_tr)
  mean_buckets_ctr=np.mean(buckets_means_ctr)
  
  
  degrees_freedom = len(means_diff) - 1
  sample_mean = np.mean(means_diff)
  sample_standard_error = stats.sem(means_diff)
  confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
  ci_low=confidence_interval[0]
  ci_up=confidence_interval[1]
  ci_prc_low=ci_low/mean_buckets_ctr
  ci_prc_up=ci_up/mean_buckets_ctr
  
   
  return np.round(pval_mean,4),np.round(mean_buckets_ctr,4),np.round(mean_buckets_tr,4),np.round(relative_diff(mean_buckets_tr,mean_buckets_ctr), 5), np.round(ci_prc_low,5), np.round(ci_prc_up,5) 


def get_bootstrap_results_for_conversions_mde(control_num_arr, control_denum_arr, uplift=0.0, bootstrap_iterations=800):
  bootstrap_sample_size=len(control_denum_arr)
  boot_means_ctr=[]
  boot_means_tr=[]
  boot_mean_diff=[]
  k=0
  for b in np.arange(bootstrap_iterations):
    idx_control=np.random.choice(np.arange(len(control_denum_arr)), size=bootstrap_sample_size, replace=True)
    control_denum=np.sum(control_denum_arr[idx_control])
    control_num=np.sum(control_num_arr[idx_control])
    idx_tr=np.random.choice(np.arange(len(control_denum_arr)), size=bootstrap_sample_size, replace=True)
    tr_denum=np.sum(control_denum_arr[idx_tr])
    tr_num=np.sum(control_num_arr[idx_tr])
    boot_mean_ctr=control_num/control_denum
    boot_mean_tr=tr_num/tr_denum*(1+uplift)
    diff_means = boot_mean_tr - boot_mean_ctr
    if diff_means<0:
      k=k+1
  pval=2 * np.minimum(k, bootstrap_iterations - k) / bootstrap_iterations
  return np.round(pval,4)
  
def get_mde(mean_diff, ci_low, ci_up):
  if mean_diff>=0:
    mde = mean_diff - ci_low
  else:
    mde = ci_up - mean_diff
  return np.round(mde,3)


def get_metric_name(num_col, denum_col):
  if 'num_purchase_attempts_reached_payment_selected_with_rendered' in num_col and 'num_purchase_attempts_with_payment_methods_rendered' in denum_col:
      prefix, suffix = denum_col.split('num_purchase_attempts_with_payment_methods_rendered')
      return prefix+'PSSR' + suffix
  if 'zps_kpi' in num_col and 'zps_kpi' in denum_col:
    prefix, num_col_name = num_col.split('zps_kpi_')
    denum_col_name = denum_col.split('zps_kpi_')[1]
    if num_col_name == 'payment_exits' and denum_col_name == 'num_purchase_attempts':
      return prefix+'zps_kpi_PER'
    if num_col_name == 'gmvbc' and denum_col_name == 'num_orders_placed':
      return prefix+'zps_kpi_basket_size'
  if 'num_orders' in num_col:
    num_col_suffix = num_col.split('num_orders')[1]
    if  'num_purchase_attempts_reached_payment_initiated' in denum_col and denum_col.split('num_purchase_attempts_reached_payment_initiated')[1] in num_col_suffix:
      prefix, suffix = denum_col.split('num_purchase_attempts_reached_payment_initiated')
      return prefix+'PPSR'+suffix
    if 'num_purchase_attempts' in denum_col and denum_col.split('num_purchase_attempts')[1] in num_col_suffix:
      prefix, suffix = denum_col.split('num_purchase_attempts')
      return prefix+'PACR'+suffix
    if 'num_orders_payment_initiated' in denum_col and denum_col.split('num_orders_payment_initiated')[1] in num_col_suffix:
      prefix, suffix = denum_col.split('num_orders_payment_initiated')
      if prefix == 'checkout_funnel_kpi_pisr_':
        prefix = 'checkout_funnel_'
      return prefix+'PISR'+suffix
    if denum_col == 'checkout_funnel_kpi_pisr_num_orders_placed' and num_col.startswith('checkout_funnel_kpi_pisr_num_orders_placed_'):
      return 'checkout_funnel_'+num_col.split('num_')[1]+'_share'
  if 'purchase_attempts_with_payment_methods_rendered' in num_col and denum_col==num_col.replace('_with_payment_methods_rendered', ''):
    prefix, suffix = num_col.split('purchase_attempts_with_payment_methods_rendered')
    if prefix == 'checkout_funnel_kpi_':
      prefix = 'checkout_funnel_'
    return prefix + 'PSVR' + suffix
  
  patterns = {'purchase_attempts_reached_payment_selected': '2_ReachedPaymentSelected'

              }
  
  for num_col_pattern, name in patterns.items():
    if num_col_pattern in num_col and 'purchase_attempts' in denum_col and denum_col.endswith(num_col.split(num_col_pattern)[1]):
      prefix, suffix = num_col.split(num_col_pattern)
      if prefix == 'checkout_funnel_kpi_':
        prefix = 'checkout_funnel_'
      return prefix+name+suffix 
  
  if 'num_authorizations_approved' in num_col and 'num_authorizations' in denum_col and denum_col.endswith(num_col.split('num_authorizations_approved')[1]):
      prefix, suffix = num_col.split('num_authorizations_approved')
      return prefix+'authorization_rate'+suffix
    
  
  if 'num_authentications_approved' in num_col and 'num_authentications' in denum_col and denum_col.endswith(num_col.split('num_authentications_approved')[1]):
      prefix, suffix = num_col.split('num_authentications_approved')
      return prefix+'authentication_rate'+suffix
    
  if num_col.startswith('friction_') and num_col.startswith(denum_col):
    return num_col+'_share'
  if num_col == 'friction_successful_pi_time_sum' and denum_col == 'friction_succesful_pa':
    return 'seconds_between_order_confirmation_and_placement'
  if num_col == 'friction_pi_time_sum' and denum_col == 'friction_pi_time_denum':
    return 'seconds_between_order_confirmation_and_placement_incl_unsuccesful'

  if num_col == 'zps_cuca_case_customers' and denum_col == 'zps_all_pa_customers':
    return 'zps_supporting_kpi_CuCa_customers'
    return 'checkout_funnel_PER'
  elif num_col=='num_orders_deferred' and denum_col=='num_sessions_offered_deferred':
    return 'deferred_adoption_rate'
  return (num_col+'/'+denum_col)


def add_new_values_to_dict(main_dict, new_values):
  for k,v in new_values.items():
    if k not in main_dict:
      main_dict[k] = v
      continue
    if type(main_dict[k]) == str:
      main_dict[k] = [main_dict[k]]
    if type(v) == str:
      v = [v]
    for v_elem in v:
      if v_elem not in main_dict[k]:
        main_dict[k].append(v_elem)

# COMMAND ----------

def toPandas_with_cuped(df_customer, quantile_for_outliers=0.9999):
  pre_columns = [x for x in df_customer.columns if x.startswith('pre_')]
  use_cuped = len(pre_columns) > 0
  if use_cuped:
    orig_columns = ['_'.join(x.split('_')[2:]) for x in pre_columns]
    best_pairs_dict = {'adj_' + x: [x, pre_x] 
                        for pre_x, x in zip(pre_columns, orig_columns)
                      }
    df_customer = winsorize(
        df = df_customer,
        specs = {x: (0, quantile_for_outliers) for x in pre_columns+orig_columns},
        prefix = "")
    adjustor_inst_null = simple_OLS(
        df = df_customer,
        column_names = best_pairs_dict
        )
    df_customer = apply_the_adjuster_and_readjustment(
        df = df_customer,
        adjuster = adjustor_inst_null,
        per_user_segmentation_to_guide_the_adjustments = [],
        ).cache()
    df_customer = coalesce_nulls_with_nonadj_value(df = df_customer, metrics = orig_columns, suffix = '')
    df_customer = df_customer.drop(*pre_columns)
  df_customer = df_customer.toPandas()
  if use_cuped:
    df_customer=df_customer.rename(columns={'adj_'+x: 'cuped_'+x for x in orig_columns})
  return df_customer


# returns:
# 1. data frame with joined data from query, query_flat and zps_kpis (if flag is present) 
# 2. updated list of metrics_columns accordingly to query_flat and zps_kpis
def get_data_for_ab_test_calculation(customer_id_column, test_start_date, day
                                     , metrics_columns, query
                                     , query_flat=None
                                     , frequency_sales_channel=None
                                     , checkout_funnel_config=None
                                     , experiment_id=None
                                     , quantile_for_outliers=0.9999
                                     , zps_kpis_mode=WITH_KPI
                                     , assignment_key='customer_id'
                                     , prewritten_metric_groups=[]
                                     , precalculated_metrics_dictionary={}
                                     ):
  
  metrics_columns_internal = [x for x in metrics_columns]
  is_zps_kpis_list_needed = zps_kpis_mode != NO_KPI

  df_customer = dwh_read(query)
  df_customer = toPandas_with_cuped(df_customer, quantile_for_outliers)

# calculate columns as per the dictionary configuration
  if precalculated_metrics_dictionary is not None:
    for metric_group in precalculated_metrics_dictionary:
      metric_group_config = precalculated_metrics_dictionary[metric_group]

      dictionary_query=dictionary_with_functions[metric_group](start_date=test_start_date, end_date=day, experiment_id=experiment_id, configuration=metric_group_config, assignment_key=assignment_key)
      df_dictionary = dwh_read(dictionary_query)
      df_dictionary = df_dictionary.toPandas()

      df_customer = df_customer.merge(df_dictionary, left_on=customer_id_column, right_on='customer_id', how='left')
      for column in df_dictionary.columns.tolist():
        df_customer[column] = df_customer[column].fillna(0)
      df_dictionary = None

  
  
#   calculate additional columns: ZPS KPIs and freq related metrics
  if is_zps_kpis_list_needed:
    zps_kpis_query=get_zps_kpis_query(start_date=test_start_date, end_date=day, experiment_id=experiment_id, with_pre=zps_kpis_mode in (KPI_WITH_CUPED, ONLY_CUPED_KPI), assignment_key=assignment_key)
    df_zps_kpis = dwh_read(zps_kpis_query)
    if zps_kpis_mode in (KPI_WITH_CUPED, ONLY_CUPED_KPI):
      no_cuped_list = ['customer_id', 'cuca_case_customers', 'all_pa_customers']
      filter_pre = [x for x in df_zps_kpis.columns 
                    if x.startswith('pre') and any(x.endswith(y) for y in no_cuped_list)]
      df_zps_kpis = df_zps_kpis.drop(*filter_pre)

    df_zps_kpis = toPandas_with_cuped(df_zps_kpis, quantile_for_outliers)

    df_customer = df_customer.merge(df_zps_kpis, left_on=customer_id_column, right_on='customer_id', how='left')
    for column in df_zps_kpis.columns.tolist():
      df_customer[column] = df_customer[column].fillna(0)
    df_zps_kpis = None

    for kpi in zpi_kpis_lists(type='metric', mode=zps_kpis_mode):
      metrics_columns_internal.append(kpi)
    

  if frequency_sales_channel is not None:
    customer_frequency_query=get_customer_frequency_query(start_date=test_start_date, end_date=day, sales_channel=frequency_sales_channel)
    df_customer_frequency = dwh_read(customer_frequency_query)
    df_customer_frequency = df_customer_frequency.toPandas()
    
    df_customer = df_customer.merge(df_customer_frequency, left_on=customer_id_column, right_on='customer_id', how='left')
    df_customer_frequency = None
    df_customer['purchase_attempt_frequency'] = df_customer['purchase_attempt_frequency'].fillna('low')


  
#   calculate additional metrics from the query flat
  if query_flat:
    df_customer_flat = dwh_read(query_flat).toPandas()
      
    flat_metrics = df_customer_flat.metric_name.unique().tolist()
    main_metrics = df_customer.columns.tolist()
    
    metrics_columns_internal += flat_metrics

    
    df_customer[customer_id_column]=df_customer[customer_id_column].astype(str)
    df_customer_flat[customer_id_column]=df_customer_flat[customer_id_column].astype(str)
    
    for metric in flat_metrics:
      if metric in main_metrics:
          continue
      res = df_customer_flat[df_customer_flat.metric_name == metric][[customer_id_column, 'metric_value']].rename(columns={'metric_value': metric})
      df_customer = df_customer.merge(res, on=customer_id_column, how='left')
    df_customer_flat = None
  
  return df_customer, metrics_columns_internal


# COMMAND ----------

def result_template(**kwargs):
  return pd.Series(data={k: kwargs.get(k, np.NaN) for k in (
    'experiment_name'
    ,'dt'
    ,'metric'
    ,'metric_type'
    ,'mean_control'
    ,'mean_treatment'
    ,'mean_diff'
    ,'mean_control_wo_outliers'
    ,'mean_treatment_wo_outliers'
    ,'mean_diff_wo_outliers'
    ,'median_control'
    ,'median_treatment'
    ,'median_diff'
    ,'ttest_pval'
    ,'ttest_mean_diff'
    ,'ttest_ci_prc_low'
    ,'ttest_ci_prc_up'
    ,'ttest_wo_outliers_pval'
    ,'ttest_wo_outliers_mean_diff'
    ,'ttest_wo_outliers_ci_prc_low'
    ,'ttest_wo_outliers_ci_prc_up'
    ,'btstrp_pval'
    ,'btstrp_mean_diff'
    ,'btstrp_ci_prc_low'
    ,'btstrp_ci_prc_up'
    ,'btstrp_wo_outliers_pval'
    ,'btstrp_wo_outliers_mean_diff'
    ,'btstrp_wo_outliers_ci_prc_low'
    ,'btstrp_wo_outliers_ci_prc_up'
    ,'buckets_pval'
    ,'buckets_mean_diff'
    ,'buckets_ci_prc_low'
    ,'buckets_ci_prc_up'
    ,'buckets_wo_outliers_pval'
    ,'buckets_wo_outliers_mean_diff'
    ,'buckets_wo_outliers_ci_prc_low'
    ,'buckets_wo_outliers_ci_prc_up'
    ,'MW_pval'
    ,'chi2_pval'
    ,'outliers_percentile'
    ,'outliers_threshold'
    ,'calculations_time'
    ,'mde_bootstrap'
    ,'mde_ttest_wo_outliers'
    ,'units_control'
    ,'units_treatment'
    ,'srm_pval'
    ,'notebook_url'
  )},name=kwargs['calculations_time'])

# COMMAND ----------

# returns dataframe with staticall analysis for the data provided in the df_customer
def get_tests_for_ab_test_calculation(df_customer, metrics_columns_internal
                                      , day
                                      , experiment_name
                                      , need_bootstrap, need_buckets
                                      , need_boostrap_wo_outliers, need_buckets_wo_outliers
                                      , variant_col, control_gr_name, treatment_gr_name
                                      , quantile_for_outliers
                                      , bootstrap_iterations
                                      , buckets_num
                                      , proportions_metrics_flag=False, proportions_columns={}
                                      , ratio_metrics_flag=False, ratio_columns={}
                                      , start_time=None
                                      , calculations_time=None
                                      , need_full_logs=True
                                      , gr_ratio=1.0
                                      ):
  
  if start_time is None:
    start_time = time.time()
  if calculations_time is None:
    calculations_time = pd.to_datetime("today").strftime('%Y-%m-%d-%H-%M-%S')

  need_wo_outliers = True
  if not need_boostrap_wo_outliers and not need_buckets_wo_outliers:
    need_wo_outliers = False

  df_results=pd.DataFrame()
    
  for metric in metrics_columns_internal:
    is_median_needed=False
    metric_type = 'none'
    if type(metric) is dict:
      metric_name=list(metric.keys())[0]
      metric_type=metric.get(metric_name)
      metric=metric_name
      print (metric_name,metric_type,'start calculations' )
    else: 
      print(metric,'start calculations')
    metric_stat = {
      'experiment_name':experiment_name
      ,'dt':day
      ,'metric': metric
      ,'metric_type': 'continuous'
      ,'outliers_percentile':quantile_for_outliers
      ,'calculations_time':calculations_time
      }
    
    if metric_type.lower() == 'median':
      is_median_needed=True
    #     remove outliers 
    if need_wo_outliers:
      df_wo_outliers, metric_stat['outliers_threshold'] = remove_outliers(df_customer, metric, quantile_for_outliers)

    #     get needed data from DF
    control,treatment = get_control_and_treatment_samples(df_customer,metric,variant_col,control_gr_name,treatment_gr_name)

    metric_stat['units_control'] = len(control)
    metric_stat['units_treatment'] = len(treatment)
    metric_stat['srm_pval'] = np.round(stats.chisquare([len(control),len(treatment)*gr_ratio])[1],4)

  #  Mean, median and its differences
    metric_stat['mean_control']=np.round(control.mean(),4)
    metric_stat['mean_treatment']=np.round(treatment.mean(),4)
    metric_stat['mean_diff']=np.round(relative_diff(metric_stat['mean_treatment'],metric_stat['mean_control']),4)
    
    if not is_median_needed:
    #     T-test pvalue calculations
      metric_stat['ttest_pval'], _, _, metric_stat['ttest_mean_diff'], metric_stat['ttest_ci_prc_low'], metric_stat['ttest_ci_prc_up'] = get_t_test_results(control,treatment)
      if need_wo_outliers:
        control_wo_outliers,treatment_wo_outliers = get_control_and_treatment_samples(df_wo_outliers,metric,variant_col,control_gr_name,treatment_gr_name)
        metric_stat['ttest_wo_outliers_pval'], _, _, metric_stat['ttest_wo_outliers_mean_diff'], metric_stat['ttest_wo_outliers_ci_prc_low'], metric_stat['ttest_wo_outliers_ci_prc_up'] = get_t_test_results(control_wo_outliers,treatment_wo_outliers)
        metric_stat['mean_control_wo_outliers']=np.round(control_wo_outliers.mean(),4)
        metric_stat['mean_treatment_wo_outliers']=np.round(treatment_wo_outliers.mean(),4)
        metric_stat['mean_diff_wo_outliers']=np.round(relative_diff(metric_stat['mean_treatment_wo_outliers'],metric_stat['mean_control_wo_outliers']),4)
        metric_stat['mde_ttest_wo_outliers'] = get_mde(metric_stat['mean_diff'], metric_stat['ttest_ci_prc_low'], metric_stat['ttest_ci_prc_up'])

    #  MW test calculations
      metric_stat['MW_pval']=stats.mannwhitneyu(control,treatment).pvalue
      

    metric_stat['median_control']=np.median(control)
    metric_stat['median_treatment']=np.median(treatment)
    metric_stat['median_diff']=np.round(relative_diff(metric_stat['median_treatment'],metric_stat['median_control']),4)

#     #     CI bounds calculation
#     ci_diff_primary = get_z_test_CI(control,treatment)
#     ci_diff_primary_wo_outliers = get_z_test_CI(control_wo_outliers,treatment_wo_outliers)

    if need_full_logs:
      print(" %s minutes needed for usual tests calculations " % (np.round((time.time() - start_time)/60,1)))

    # bootstrap
    if need_bootstrap:
        metric_stat['btstrp_pval'], _, _, metric_stat['btstrp_mean_diff'], metric_stat['btstrp_ci_prc_low'], metric_stat['btstrp_ci_prc_up'] = get_bootstrap_results(control,treatment,bootstrap_iterations, bootstrap_median = is_median_needed)
        metric_stat['mde_bootstrap'] = get_mde(metric_stat['btstrp_mean_diff'], metric_stat['btstrp_ci_prc_low'], metric_stat['btstrp_ci_prc_up'])

    if need_boostrap_wo_outliers and not is_median_needed:
      metric_stat['btstrp_wo_outliers_pval'], _, _, metric_stat['btstrp_wo_outliers_mean_diff'], metric_stat['btstrp_wo_outliers_ci_prc_low'], metric_stat['btstrp_wo_outliers_ci_prc_up'] = get_bootstrap_results(control_wo_outliers,treatment_wo_outliers,bootstrap_iterations)

    if (need_bootstrap or (need_boostrap_wo_outliers and not is_median_needed)) and need_full_logs:
      print(" %s minutes needed for usual+bootstrap tests calculations " % (np.round((time.time() - start_time)/60,1)))

    # buckets+t_test
    if need_buckets:
      metric_stat['buckets_pval'], _, _, metric_stat['buckets_mean_diff'], metric_stat['buckets_ci_prc_low'], metric_stat['buckets_ci_prc_up'] = get_buckets_results(control,treatment,buckets_num)

    if need_buckets_wo_outliers:
      metric_stat['buckets_wo_outliers_pval'], _, _, metric_stat['buckets_wo_outliers_mean_diff'], metric_stat['buckets_wo_outliers_ci_prc_low'], metric_stat['buckets_wo_outliers_ci_prc_up'] = get_buckets_results(control_wo_outliers,treatment_wo_outliers,buckets_num)

    if (need_buckets or need_buckets_wo_outliers) and need_full_logs:
      print(" %s minutes needed for usual+bootstrap+buckets tests calculations " % (np.round((time.time() - start_time)/60,1)))  

    #     collect results and append it to results df
    if is_median_needed:
      metric_stat['metric'] = metric + '_median'
      metric_stat['mean_control'] = metric_stat['median_control']
      metric_stat['mean_treatment'] = metric_stat['median_treatment']
      metric_stat['mean_diff'] = metric_stat['median_diff']
           
    df_results=df_results.append(result_template(**metric_stat), ignore_index=False)
    
  if proportions_metrics_flag:
    for num_col in proportions_columns.keys():
      print(num_col,'proportion start calculations')
      if type(proportions_columns[num_col]) == str:
        proportions_columns[num_col] = [proportions_columns[num_col]]
      for denum_col in proportions_columns[num_col]:
        metric_stat = {
          'experiment_name':experiment_name
          ,'dt':day
          ,'metric': get_metric_name(num_col, denum_col)
          ,'metric_type': 'proportion'
          ,'calculations_time':calculations_time
          }
        control_num,treatment_num, control_denum,treatment_denum = get_control_and_treatment_samples_num_denum(df_customer,num_col,denum_col,variant_col,control_gr_name,treatment_gr_name)
        metric_stat['chi2_pval'], ci_low, ci_up, metric_stat['btstrp_ci_prc_low'], metric_stat['btstrp_ci_prc_up'], metric_stat['mean_control'], metric_stat['mean_treatment'], p_diff, metric_stat['mean_diff'] = get_chi2_test_results(control_num,control_denum,treatment_num,treatment_denum)

        metric_stat['mde_bootstrap'] = get_mde(metric_stat['mean_diff'], metric_stat['btstrp_ci_prc_low'], metric_stat['btstrp_ci_prc_up'])
        
        metric_stat['units_control'] = np.sum(control_denum)
        metric_stat['units_treatment'] = np.sum(treatment_denum)
        metric_stat['srm_pval'] = np.NaN

        df_results=df_results.append(result_template(**metric_stat), ignore_index=False)
        if need_full_logs:
          print(" %s minutes needed for proportions usual tests calculations " % (np.round((time.time() - start_time)/60,1)))
    
  if ratio_metrics_flag:
    for num_col in ratio_columns.keys():
      print(num_col,'ratio start calculations')
      if type(ratio_columns[num_col]) == str:
        ratio_columns[num_col] = [ratio_columns[num_col]]
      for denum_col in ratio_columns[num_col]:
        metric_stat = {
          'experiment_name':experiment_name
          ,'dt':day
          ,'metric': get_metric_name(num_col, denum_col)
          ,'metric_type': 'ratio'
          ,'calculations_time':calculations_time
          }
        
        control_num,treatment_num, control_denum,treatment_denum = get_control_and_treatment_samples_num_denum(df_customer,num_col,denum_col,variant_col,control_gr_name,treatment_gr_name)
        metric_stat['units_control'] = np.sum(control_denum)
        metric_stat['units_treatment'] = np.sum(treatment_denum)
        metric_stat['srm_pval'] = np.NaN

        metric_stat['mean_control']=np.round(np.sum(control_num)/np.sum(control_denum),4)
        metric_stat['mean_treatment']=np.round(np.sum(treatment_num)/np.sum(treatment_denum),4)
        metric_stat['mean_diff']= np.round(relative_diff(metric_stat['mean_treatment'],metric_stat['mean_control']),4)
        metric_stat['btstrp_pval'], _, _, metric_stat['btstrp_mean_diff'], metric_stat['btstrp_ci_prc_low'], metric_stat['btstrp_ci_prc_up'] = get_bootstrap_results_for_conversions(control_num, control_denum, treatment_num, treatment_denum, bootstrap_iterations)
        metric_stat['mde_bootstrap'] = get_mde(metric_stat['mean_diff'], metric_stat['btstrp_ci_prc_low'], metric_stat['btstrp_ci_prc_up'])
        if need_full_logs:
          print(" %s minutes needed for ratio bootstrap tests calculations " % (np.round((time.time() - start_time)/60,1)))

        #  bucket test if requested
        if need_buckets:
          metric_stat['buckets_pval'], _, _, metric_stat['buckets_mean_diff'], metric_stat['buckets_ci_prc_low'], metric_stat['buckets_ci_prc_up'] = get_buckets_results_for_conversions(control_num, control_denum, treatment_num, treatment_denum, buckets_num)
          if need_full_logs:
            print(" %s minutes needed for ratio bootstrap+buckets tests calculations " % (np.round((time.time() - start_time)/60,1)))

        df_results=df_results.append(result_template(**metric_stat), ignore_index=False)

  return df_results

# COMMAND ----------

# runs query to get data and returns dataframe with staticall analysis for the data provided in the df_customer
def main_ab_test_calculation_function( customer_id_column, test_start_date, day, calc_start_date, calc_end_date
                                      , metrics_columns, need_bootstrap, proportions_columns, ratio_columns
                                      , query 
                                      , dimensions_columns=[]
                                      , is_zps_kpis_list_needed=True # if you want to get ZPS KPIs (GMV,PACR,PSSR,PPSR) calculated
                                      , query_flat=None
                                      , need_buckets=False, need_buckets_wo_outliers=False, buckets_num=150
                                      , proportions_metrics_flag=None, ratio_metrics_flag=None
                                      , need_full_logs=True
                                      , quantile_for_outliers=0.99999 # quantile for outliers removing. Users who have metric value above this quentile will be removed for 'without outliers' calculations
                                      , bootstrap_iterations=1000 # number of iterations in bootstrap
                                      , gr_ratio=1.0
                                      , frequency_sales_channel=None
                                      , checkout_funnel_channel=None
                                      , experiment_id=None
                                      , zps_kpis_mode=None
                                      , checkout_funnel_config=None
                                      , assignment_key='customer_id'
                                      , prewritten_metric_groups = []
                                      , precalculated_metrics_dictionary = {}
                                      ):
    
  print(day)
  start_time = time.time()

  if zps_kpis_mode is None:
    zps_kpis_mode = WITH_KPI if is_zps_kpis_list_needed else NO_KPI
  is_zps_kpis_list_needed = zps_kpis_mode != NO_KPI

  if checkout_funnel_config is None and checkout_funnel_channel is not None:
    checkout_funnel_config = {
      'channel': checkout_funnel_channel
      , 'pisr_payment_methods': [] 
      }
  if checkout_funnel_config is not None and type(checkout_funnel_config['pisr_payment_methods']) == str:
    checkout_funnel_config['pisr_payment_methods']=[checkout_funnel_config['pisr_payment_methods']]

  if not precalculated_metrics_dictionary:
    precalculated_metrics_dictionary = {}
    if checkout_funnel_config is not None:
        precalculated_metrics_dictionary['checkout_funnel_kpis'] = checkout_funnel_config
    
    if prewritten_metric_groups is not None:
        precalculated_metrics_dictionary['checkout_friction_metrics'] = prewritten_metric_groups

  df_customer, metrics_columns_internal = get_data_for_ab_test_calculation(
    customer_id_column=customer_id_column
    , test_start_date=test_start_date
    , day=day
    , metrics_columns=metrics_columns
    , query=query
    , query_flat=query_flat
    , frequency_sales_channel=frequency_sales_channel
    , experiment_id=experiment_id
    , quantile_for_outliers=quantile_for_outliers
    , zps_kpis_mode=zps_kpis_mode
    , assignment_key=assignment_key
    , precalculated_metrics_dictionary=precalculated_metrics_dictionary
  )
  
  print (list(df_customer.columns))
  print(" %s minutes needed for Query calculations " % (np.round((time.time() - start_time)/60,1)))
  start_time = time.time()

  if proportions_metrics_flag is None:
    proportions_metrics_flag = True if proportions_columns else False
  if ratio_metrics_flag is None:
    ratio_metrics_flag = True if ratio_columns else False

  if is_zps_kpis_list_needed:
    if not ratio_metrics_flag:
      ratio_columns = {}
      ratio_metrics_flag = True
    add_new_values_to_dict(ratio_columns, zpi_kpis_lists(type='ratio', mode=zps_kpis_mode))
    if not proportions_metrics_flag:
      proportions_columns = {}
      proportions_metrics_flag = True
    add_new_values_to_dict(proportions_columns, zpi_kpis_lists(type='proportion', mode=zps_kpis_mode))
    
  if 'checkout_funnel_kpis' in precalculated_metrics_dictionary:
    if not ratio_metrics_flag:
      ratio_columns = {}
      ratio_metrics_flag = True
    add_new_values_to_dict(ratio_columns, checkout_funnel_lists(type='ratio', configuration=precalculated_metrics_dictionary['checkout_funnel_kpis']))

  if 'checkout_friction_metrics' in precalculated_metrics_dictionary:
    checkout_friction_mode = [x for x in precalculated_metrics_dictionary['checkout_friction_metrics'] if x in (CHECKOUT_FRICTION, CHECKOUT_FRICTION_SUCCESSFUL_PA_ONLY)]
    if len(checkout_friction_mode) > 0:
      if not ratio_metrics_flag:
        ratio_columns = {}
        ratio_metrics_flag = True
      add_new_values_to_dict(ratio_columns, checkout_friction_lists(type='ratio', configuration=checkout_friction_mode))
  

  df_results = get_tests_for_ab_test_calculation(
    day=day
    , experiment_name=experiment_name
    , need_bootstrap=need_bootstrap
    , need_buckets=need_buckets
    , need_boostrap_wo_outliers=need_boostrap_wo_outliers
    , need_buckets_wo_outliers=need_buckets_wo_outliers
    , variant_col=variant_col
    , control_gr_name=control_gr_name
    , treatment_gr_name=treatment_gr_name
    , quantile_for_outliers=quantile_for_outliers
    , bootstrap_iterations=bootstrap_iterations
    , buckets_num=buckets_num
    , calculations_time=calculations_time
    , proportions_metrics_flag=proportions_metrics_flag
    , proportions_columns=proportions_columns
    , ratio_metrics_flag=ratio_metrics_flag
    , ratio_columns=ratio_columns
    , df_customer=df_customer
    , metrics_columns_internal=metrics_columns_internal
    , start_time=start_time
    , need_full_logs=need_full_logs
    , gr_ratio=gr_ratio
  )
  
  #dimension calculations
  if 'purchase_attempt_frequency' not in dimensions_columns and frequency_sales_channel is not None:
    dimensions_columns.append('purchase_attempt_frequency')
  available_columns = df_customer.columns.tolist()
  for column in dimensions_columns:
    if column not in available_columns:
      print('No column', column, 'in provided data')
      print('Available columns:')
      print(available_columns)
      continue
    dimension_values = df_customer[column].unique().tolist()
    print('dimension', column, 'values', dimension_values)
    for dimension in dimension_values:
      if dimension is None:
        print('Ignoring None values in the', column)
        continue
      print('Calculating', column, '=', dimension)
      df_customer_dimension = df_customer[df_customer[column] == dimension]
      experiment_name_dimension = experiment_name + ' | ' + column + ' | ' + dimension
      df_results_dimension = get_tests_for_ab_test_calculation(
        day=day
        , experiment_name=experiment_name_dimension
        , need_bootstrap=need_bootstrap
        , need_buckets=need_buckets
        , need_boostrap_wo_outliers=need_boostrap_wo_outliers
        , need_buckets_wo_outliers=need_buckets_wo_outliers
        , variant_col=variant_col
        , control_gr_name=control_gr_name
        , treatment_gr_name=treatment_gr_name
        , quantile_for_outliers=quantile_for_outliers
        , bootstrap_iterations=bootstrap_iterations
        , buckets_num=buckets_num
        , calculations_time=calculations_time
        , proportions_metrics_flag=proportions_metrics_flag
        , proportions_columns=proportions_columns
        , ratio_metrics_flag=ratio_metrics_flag
        , ratio_columns=ratio_columns
        , df_customer=df_customer_dimension
        , metrics_columns_internal=metrics_columns_internal
        , start_time=start_time
        , need_full_logs=need_full_logs
        , gr_ratio=gr_ratio
      ) 
      df_results=df_results.append(df_results_dimension, ignore_index=False)

  print(" %s minutes needed for Tests calculations " % (np.round((time.time() - start_time) / 60, 1)))
  print(day, '-done')
  return df_results
    


# COMMAND ----------

def main_function_for_sample_size_calculations(query, mde, w, treatment_share_percent, pvals, pval_counter, proportions_columns, ratio_columns, continuous_metric_name, continuous_metric_flag, proportions_metric_flag, ratio_metric_flag, continuous_metric_is_median = False
                                               , bootstrap_iterations=800):
    
  df_customer, _ = get_data_for_ab_test_calculation(
    customer_id_column=None
    , test_start_date=None
    , day=None
    , metrics_columns=[]
    , query=query
    , query_flat=None
    , frequency_sales_channel=None
    , checkout_funnel_config=None
    , experiment_id=None
    , quantile_for_outliers=0.9999 
    , zps_kpis_mode=NO_KPI
    , assignment_key='customer_id'
  )
  
  df_customer= df_customer.sample(frac=treatment_share_percent, replace=False, random_state=1)
  s_size=len(df_customer)

  
  if continuous_metric_flag:
    control=df_customer[continuous_metric_name].values
    control_uplifted=control*(1+mde)
    pval, btstrp_ctr_mean_uplifted, btstrp_tr_mean_uplifted, btstrp_mean_diff_uplifted, btstrp_ci_prc_low_uplifted, btstrp_ci_prc_up_uplifted = get_bootstrap_results(control,control_uplifted,bootstrap_iterations, bootstrap_median = continuous_metric_is_median) 

  elif proportions_metric_flag:
    if len(proportions_columns)>1:
      raise ValueError('You should input only one proportion metric')
    num_name=list(proportions_columns.keys())[0]
    denum_name=proportions_columns[num_name]
    control_num=df_customer[num_name].values
    control_denum=df_customer[denum_name].values
    treatment_uplifted_num=np.sum(control_num)*(1+mde)
    treatment_uplifted_denum=np.sum(control_denum)
    T = np.array([[np.sum(control_num), np.sum(control_denum)-np.sum(control_num)], [treatment_uplifted_num, treatment_uplifted_denum-treatment_uplifted_num]])
    pval=np.round(stats.chi2_contingency(T,correction=False)[1],5)
  
  elif ratio_metric_flag:
    if len(ratio_columns) > 1:
      raise ValueError('You should input only one ratio metric')
    
    num_name=list(ratio_columns.keys())[0]
    denum_name=ratio_columns[num_name]
    control_num=df_customer[num_name].values
    control_denum=df_customer[denum_name].values
    pval= get_bootstrap_results_for_conversions_mde(control_num, control_denum, mde, bootstrap_iterations)

  else:
    raise ValueError('You should input one metric_flag=True')

  pvals.update({w: pval})
  if pval < 0.05:
    pval_counter += 1
  else:
    pval_counter = 0
    
  print ('week=',w,'; pval=',pval)
  return pval, w, pval_counter, s_size, pvals

    


# COMMAND ----------

def main_function_for_mde_calculations(query, treatment_share_percent, proportions_columns, ratio_columns, continuous_metric_name, continuous_metric_flag, proportions_metric_flag, ratio_metric_flag, continuous_metric_is_median = False
                                               , bootstrap_iterations=2400, binary_search_depth=10):
    
  df_customer, _ = get_data_for_ab_test_calculation(
    customer_id_column=None
    , test_start_date=None
    , day=None
    , metrics_columns=[]
    , query=query
    , query_flat=None
    , frequency_sales_channel=None
    , checkout_funnel_config=None
    , experiment_id=None
    , quantile_for_outliers=0.9999 
    , zps_kpis_mode=NO_KPI
    , assignment_key='customer_id'
  )
  
  df_customer= df_customer.sample(frac=treatment_share_percent, replace=False, random_state=1)
  s_size=len(df_customer)

  iter = 0
  mde_up = 1
  mde_low = 0
  mde = 0.5
  while iter < binary_search_depth:
    
    if continuous_metric_flag:
      control=df_customer[continuous_metric_name].values
      control_uplifted=control*(1+mde)
      pval, btstrp_ctr_mean_uplifted, btstrp_tr_mean_uplifted, btstrp_mean_diff_uplifted, btstrp_ci_prc_low_uplifted, btstrp_ci_prc_up_uplifted = get_bootstrap_results(control,control_uplifted,bootstrap_iterations, bootstrap_median = continuous_metric_is_median) 

    elif proportions_metric_flag:
      if len(proportions_columns)>1:
        raise ValueError('You should input only one proportion metric')
      num_name=list(proportions_columns.keys())[0]
      denum_name=proportions_columns[num_name]
      control_num=df_customer[num_name].values
      control_denum=df_customer[denum_name].values
      treatment_uplifted_num=np.sum(control_num)*(1+mde)
      treatment_uplifted_denum=np.sum(control_denum)
      T = np.array([[np.sum(control_num), np.sum(control_denum)-np.sum(control_num)], [treatment_uplifted_num, treatment_uplifted_denum-treatment_uplifted_num]])
      pval=np.round(stats.chi2_contingency(T,correction=False)[1],5)
    
    elif ratio_metric_flag:
      if len(ratio_columns) > 1:
        raise ValueError('You should input only one ratio metric')
      
      num_name=list(ratio_columns.keys())[0]
      denum_name=ratio_columns[num_name]
      control_num=df_customer[num_name].values
      control_denum=df_customer[denum_name].values
      pval= get_bootstrap_results_for_conversions_mde(control_num, control_denum, mde, bootstrap_iterations)

    else:
      raise ValueError('You should input one metric_flag=True')

    if pval <= 0.05:
      mde_up, mde, mde_low = mde, (mde_low+mde)/2, mde_low
    else:
      mde_up, mde, mde_low = mde_up, (mde_up+mde)/2, mde

    iter += 1

  return mde

    


# COMMAND ----------

def get_summary(df_results):
  data = df_results[df_results.dt==df_results.dt.max()]
  data['pval'] = data[["btstrp_pval", "chi2_pval"]].min(axis=1)
  return data[['experiment_name', 'dt','metric', 'metric_type', 'mean_diff', 'pval']].sort_values(by=['pval'])

# returns list of metrics calculated by with less than 90% of users
def get_no_full_data_continuous_metrics(df_results, experiment_name):
  data = df_results[(df_results.metric_type == 'continuous') & (df_results.dt==df_results.dt.max())& (df_results.experiment_name==experiment_name)]
  threshold = 0.9*max(data.units_control.max(), data.units_treatment.max())
  return data[(data.units_control<0.9*data.units_control.max()) | (data.units_treatment<0.9*data.units_treatment.max())].metric.unique()
