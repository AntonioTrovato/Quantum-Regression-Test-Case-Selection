import csv
import statistics

import numpy as np
import pandas as pd
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import random
import time
from dimod.serialization.format import Formatter

def get_data(data_name):
    data = pd.read_csv("./"+data_name+"/"+data_name+".csv", dtype={"time": float, "rate": float})
    data = data.drop(data[data['rate'] == 0].index)
    return data

def bootstrap_sampling(data, sample_time, sample_size):
    '''
    param data: data frame
    :param sample_time: number of sampling
    :param sample_size: number of test cases in a sample
    :return: a two-dimension list of all sampled test cases
    '''
    start = time.time()
    sample_list_total = []
    for i in range(sample_time):
        sample_list=random.sample(range(0,len(data["time"])),sample_size)
        sample_list_total.append(sample_list)
    end = time.time()
    return sample_list_total, (end-start)*1000


def create_bqm(sample, sample_size,sample_time, data):
    '''
    :param sample: a list of sampled test cases
    :param sample_time:
    :param data: dataframe
    :return: a bqm of the objective function
    '''
    dic_time = {}
    dic_rate = {}
    dic_num = {}
    time_total = 0
    rate_total = 0
    for id in sample:
        dic_time["T"+str(id)] = data["time"].iloc[id]
        time_total += data["time"].iloc[id]
        dic_rate["T"+str(id)] = data["rate"].iloc[id]
        rate_total += data["rate"].iloc[id]
        dic_num["T"+str(id)] = 1

    bqm_time = dimod.BinaryQuadraticModel(dic_time,{}, 0,dimod.Vartype.BINARY)
    bqm_rate = dimod.BinaryQuadraticModel(dic_rate,{}, 0,dimod.Vartype.BINARY)
    bqm_num = dimod.BinaryQuadraticModel(dic_num,{}, 0,dimod.Vartype.BINARY)

    bqm_time.normalize()
    bqm = (1/3)*pow((bqm_time-0)/sample_size, 2) + (1/3)*pow((bqm_rate - rate_total)/sample_size, 2) + (1/3)*pow((bqm_num - 0)/sample_size, 2)

    return bqm


def run_qpu(sample_list_total, data, sample_time, sample_size, it):
    '''
    :param sample_list_total: all sampled test cases
    :param data: dataframe
    :return: energy and sample of the best solution
    '''
    sample_first_list = []
    qpu_access_times = []

    for i in range(len(sample_list_total)):
        obj = create_bqm(sample_list_total[i], sample_size,sample_time, data)
        sampler = EmbeddingComposite(DWaveSampler(token="DEV-aed1bf4d378bc904cc1f3f2b30b10d9e16eaeb9f"))
        sampleset = sampler.sample(obj, num_reads=100, return_embedding=True)
        Formatter(sorted_by=None).fprint(sampleset)
        qpu_access = sampleset.info['timing']['qpu_access_time']
        qpu_access_times.append(qpu_access/1000)

        first_sample = sampleset.first.sample
        sample_first_list.append(first_sample)

        print("\nSample number: " + str(i) + "\nIt number: " + str(it) + "\nTot samples: " + str(len(sample_list_total)))

    avg_qpu_access_time = statistics.mean(qpu_access_times)

    return sample_first_list, avg_qpu_access_time

def gen_dic(data):
    foods = {}
    for i,x in enumerate(data[["time","rate"]].to_dict(orient="records")):
        foods["T{}".format(i)] = x
    return foods

def make_summary(sample,data):
    foods = gen_dic(data)
    total_time = 0
    total_rate = 0
    for t in sample.keys():
        if t[0] == 'T' and sample[t] == 1:
            total_time += foods[t]['time']
            total_rate += foods[t]['rate']
    return total_time, total_rate

def merge(sample_list):
    case_list = {}
    for i in range(len(sample_list)):
        for t in sample_list[i].keys():
            if t[0] == 'T' and sample_list[i][t] == 1:
                case_list[t] = 1
    return case_list

def bootstrap_confidence_interval(data, num_samples, confidence_alpha=0.95):
    """This function determines the statistical range within we would expect the mean value of execution times to fall; it relies on the bootstrapping strategy, which allows the calculation of the confidence interval by repeatedly sampling (with replacement) from the existing data to obtain an estimate of the confidence interval."""
    sample_means = []
    for _ in range(num_samples):
        bootstrap_sample = [random.choice(data) for _ in range(len(data))]
        sample_mean = np.mean(bootstrap_sample)
        sample_means.append(sample_mean)

    lower_percentile = (1 - confidence_alpha) / 2 * 100
    upper_percentile = (confidence_alpha + (1 - confidence_alpha) / 2) * 100
    lower_bound = np.percentile(sample_means, lower_percentile)
    upper_bound = np.percentile(sample_means, upper_percentile)

    return lower_bound, upper_bound

if __name__ == '__main__':
    #data_names -> list of dataset names
    #samples_params = {"data_name":(n_size,m_time),...}
    #repeat = #experiment repetitions
    #data_names = ["gsdtsr","paintcontrol"]
    data_names = ["iofrol"]
    samples_params = {"gsdtsr":(20,21),"paintcontrol":(30,6),"iofrol":(30,127)}
    repeat = 10

    for data_name in data_names:
        sampling_times = []
        qpu_access_times = []
        execution_times = []
        sol_suite_costs = []
        sol_suite_rates = []
        print(data_name)

        for x in range(10):
            print(x)
            sample_size = samples_params[data_name][0]
            sample_time = samples_params[data_name][1]
            data = get_data(data_name)
            sample_total_list, sampling_time = bootstrap_sampling(data, sample_time, sample_size)
            sample_first_list, avg_qpu_access_time = run_qpu(sample_total_list, data, sample_time, sample_size, x)
            merge_sample = merge(sample_first_list)
            sol_suite_cost, sol_suite_rate = make_summary(merge_sample, data)
            sampling_times.append(sampling_time)
            qpu_access_times.append(avg_qpu_access_time)
            execution_times.append(sampling_time + avg_qpu_access_time)
            sol_suite_costs.append(sol_suite_cost)
            sol_suite_rates.append(sol_suite_rate)

        avg_sampling_time = statistics.mean(sampling_times)
        avg_qpu_access_time = statistics.mean(qpu_access_times)
        avg_execution_time = statistics.mean(execution_times)
        print("EXECUTION TIMES: " + str(execution_times))
        print("STD DEV EXECTUTION TIMES: " + str(statistics.stdev(execution_times)))
        qpu_lower_bound, qpu_upper_bound = bootstrap_confidence_interval(qpu_access_times, 100, 0.95)

        var_names = ["final_test_suite_costs", "final_failure_rates", "avg_sampling_time(ms)",
                     "avg_qpu_access_time(ms)", "qpu_lower_bound(ms)", "qpu_upper_bound(ms)",
                     "avg_ex_time(ms)", "std_dev_ex_time", "exectution_times(ms)"]
        values = [sol_suite_costs, sol_suite_rates, avg_sampling_time,
                  avg_qpu_access_time, qpu_lower_bound, qpu_upper_bound, avg_execution_time,
                  statistics.stdev(execution_times),execution_times]

        with open("./" + data_name + "/sum_bootqa.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(var_names)
            writer.writerow(values)
