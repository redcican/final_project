from dataclasses import replace
from utils import config
from utils.transform import Input, Output, Excel, get_experiments_url, get_optimierung_url, get_predict_url, get_experiments_dataset_url, \
    get_file_url, distribution_density, get_evaluation_url, Histogram
from utils import services
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, \
    confusion_matrix, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, \
    mean_absolute_percentage_error
import numpy as np
from autogluon.tabular import TabularPredictor as task
import pandas as pd
import shap
from more_itertools import grouper
from optirodig.models import Schrott, SchrottChemi, GiessereiSchrott
from django_pandas.io import read_frame

from scipy.optimize import Bounds
from scipy.optimize import minimize
import tensorflow as tf
import xgboost as xgb
import cobyqa
from pdfo import pdfo
import time
from django.conf import settings
import os

def parse_float(val):
    try:
        if np.isnan(val):
            return None
        return float(val)
    except:
        return val

chemi_names = ['C','Si','Mn','Cr','Mo','V']

class ModelWrapper:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict(self, data):
        return np.array(self.model.predict(pd.DataFrame(data, columns=self.features)))

    def predict_prob(self, data):
        predict_value = np.array(self.model.predict_proba(pd.DataFrame(data, columns=self.features)))

        if predict_value.ndim == 1:
            predict_value = np.array([predict_value, 1 - predict_value]).T

        return predict_value

class Train:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id

        params = {"experiment_id": experiment_id}
        self.experiment_info = services.get(target_path=config.TARGET_PATH.get("experiment"), params=params)

        # Model setting
        #self.preset = 'medium_quality_faster_train'
        # self.preset = 'best_quality'
        self.time_limit = None
        self.feature_generator = config.FEATURE_GENERATOR
        self.hyper = config.HYPER_PARAMETER

    def train(self):
        # Load data
        file_url = get_file_url(self.experiment_info.get("file_id"))
        df_data = Input(file_url).from_csv()

        if config.DEBUG:
            df_data = df_data.head(100)

        # Split train test
        df_train, df_test = self.pre_process(df_data)

        # Build model
        output_path = get_experiments_url(self.experiment_id)
        presets = 'best_quality'
        predictor = task(label=self.experiment_info.get("target"),
                         path=output_path,
                         problem_type=self.experiment_info.get("problem_type"),
                         eval_metric=self.experiment_info.get("score"),
                         verbosity=1).fit(
            train_data=df_train[self.experiment_info.get("features") + [self.experiment_info.get("target")]],
            presets=presets,
            time_limit=self.time_limit,
            hyperparameters=self.hyper,
            _feature_generator_kwargs=self.feature_generator,
        )

        # Get models info
        self.models(predictor, df_test)
        
        # Update Best model
        _data = {
            "experiment_id": self.experiment_id,
            "best_model_id": self._get_model_id(predictor.get_model_best()),

        }
        services.post(data=_data, target_path=config.TARGET_PATH.get("experiment"))

        # Save test data into experiments folder
        train_url = get_experiments_dataset_url(self.experiment_id, "train.pickle")
        test_url = get_experiments_dataset_url(self.experiment_id, "test.pickle")

        Output(train_url).to_pickle(df_train)
        Output(test_url).to_pickle(df_test)

    def pre_process(self, df_data):
        # Drop Null at target
        df_data.dropna(subset=[self.experiment_info.get("target")], inplace=True)

        # Fill Null by 0
        df_data.fillna(0, inplace=True)

        # Split train
        df_train = df_data.sample(frac=float(self.experiment_info.get("split_ratio")) * 0.1)
        df_test = df_data.drop(df_train.index)

        return df_train, df_test

    def models(self, predictor, df_test):
        df_models_info = predictor.leaderboard(df_test, silent=True)

        _rename = {"pred_time_test": "predict_time", "model": "model_name"}
        df_models_info.rename(columns=_rename, inplace=True)

        # Importance features calculation
        for model_name in predictor.get_model_names():
            _f_impt = predictor.feature_importance(data=df_test, model=model_name, feature_stage='original',
                                                   num_shuffle_sets=1,
                                                   include_confidence_band=False)["importance"].astype(np.float16)

            df_models_info.loc[df_models_info.model_name == model_name, "features_importance"] = [_f_impt.to_dict()]

        df_models_info[["score_test", "score_val"]] = df_models_info[["score_test", "score_val"]].abs()
        df_models_info["model_id"] = df_models_info["model_name"].apply(lambda x: self._get_model_id(x))

        _info = ["model_id", "model_name", "score_test", "score_val", "fit_time", "predict_time", "features_importance"]
        models_list = df_models_info[_info].to_dict("records")

        # Post model list to server
        _data = {
            "experiment_id": self.experiment_id,
            "models_list": models_list
        }
        services.post(data=_data, target_path=config.TARGET_PATH.get("model"))

    def _get_model_id(self, model_name):
        return "{}_{}".format(self.experiment_id, model_name)

class Predict:
    def __init__(self, predict_id):
        self.predict_id = predict_id

        params = {"predict_id": self.predict_id}
        self.predict_info = services.get(config.TARGET_PATH.get("predict"), params=params)
        self.experiment_info = self.predict_info.get("experiment")

    def predict(self):
        # Load experiment id
        _save_dir = get_experiments_url(self.experiment_info.get("experiment_id"))
        predictor = task.load(_save_dir)

        # Load data
        file_url = get_file_url(self.predict_info.get("file_id"))
        df_predict = Input(file_url).from_csv()

        # Predict
        if self.experiment_info.get("problem_type") == "regression":
            _predict_result = predictor.predict(df_predict, self.predict_info.get("model_name"))

            df_predict["Predict"] = _predict_result
        else:
            _predict_result = predictor.predict_proba(df_predict, self.predict_info.get("model_name"))

            _rename = dict(zip(_predict_result.columns, ["predict_class_{}".format(str(label).strip()) for label in _predict_result.columns.tolist()]))

            _predict_result.rename(columns=_rename, inplace=True)

            df_predict = pd.concat([df_predict, _predict_result], axis=1)

        # Save file into local
        file_path = get_predict_url(self.predict_id)
        try:
            excel = Excel(file_path)

            excel.add_data(worksheet_name="Data", pd_data=df_predict, header_lv0=None, is_fill_color_scale=False, columns_order=None)

            excel.save()
        except Exception as e:
            mes = 'Can not generate excel file.  ERROR : {}'.format(e)
            raise Exception(mes)

class SchrottOptimierung:
    def __init__(self, optimierung_id):
        self.optimierung_id = optimierung_id
        params = {"optimierung_id": self.optimierung_id}
        self.experiment_info = services.get(config.TARGET_PATH.get("schrottoptimierung"), params=params)
        
    def optimize(self, total_quantity, chemi_component, selected_stahl_name):
        """
        Args:
            total_quantity (int): the total amount of scraps to optimize in kg 
            chemi_component (list): the list of chemical components to optimize
            selected_stahl_name (str): the name of the final steel product to optimize (1.2343, 1.2344,1.2379,1.3343)
        """
        
        # chemi component and total weight of the final steel product to optimize
        chemi_component = (np.array(chemi_component) / 100.0).astype(np.float32)
        
        total_chemi_to_achieve = total_quantity * chemi_component
        
        # load the original training dataframe, chemical dataframe and price dataframe from database
        
        # get the giesserei company names
        # giesserei_company_names = GiessereiSchrott.objects.values_list("giesserei", flat=True).distinct()
        # giesserei_company_names = sorted(list(giesserei_company_names))
        
        # Load training data
        file_url = get_file_url(self.experiment_info.get("file_id"))
        df = Input(file_url).from_csv()
        df = df[self.experiment_info.get("features")]   # extract the used features from the dataframe
        
        # convert the `Schrott` model to a pandas dataframe, and extract the `price` column
        schrott_obj = Schrott.objects.all()
        df_schrott = read_frame(schrott_obj)
        df_price = df_schrott["price"].to_numpy().astype(np.float32)# the number of schrott * the number of company
        company_count = int(df_schrott[["company"]].nunique())
        
        # load the chemical dataframe
        chemi_obj = SchrottChemi.objects.all()
        df_chemi = read_frame(chemi_obj)
        df_chemi[chemi_names] = df_chemi[chemi_names].astype(np.float32)
        
        ############################# Optimization #############################
        constant_features_names, schrotte_features_names, kreislauf_schrotte_names, legierung_schrotte_names,fremd_schrotte_names = df_columns_name(df)
        
        length_fremdschrott = len(fremd_schrotte_names) 
        total_variable_length = length_fremdschrott * company_count  # the total number of variable parameters to optimize
        
        price_list = df_price[:total_variable_length]  # the price list of all the schrott
        
        ############################## Optimization Settings ##############################
        x_lower = np.zeros(total_variable_length)  # the lower bound of the variable parameters
        x_upper = np.ones(total_variable_length) * total_quantity # the upper bound of the variable parameters
        
        fremdschrotte_chemi_table = fremdschrott_chemi_table(df_chemi,fremd_schrotte_names,company_count)
        
        # right hand side of equality constraints
        aeq = np.array(fremdschrotte_chemi_table)
        
        # bounds of scipy constraints
        bounds = Bounds([0.0]*total_variable_length, [max(total_chemi_to_achieve)]*total_variable_length)
        
        # max iteration 
        max_iter = 300
        ############################## Optimization Settings ##############################
        
        ############################## ML Settings ##############################
        xgb_model = xgb.Booster()
        # based on the selected steel name, load the corresponding ml model (1.2343, 1.2344,1.2379,1.3343)
        xgb_model.load_model(os.path.join(settings.ML_MODELS, f'{selected_stahl_name}/XGB.json'))
        ann_model = tf.keras.models.load_model(os.path.join(settings.ML_MODELS, f'{selected_stahl_name}/ANN'))
        
        # function for xgb prediction
        def f_xgb(x):
            y = xgb_model.predict(xgb.DMatrix([x]))   #x1.reshape(1,-1))
            return y.item()

        # function to calculate the total cost of xgboost version
        def sum_t3_xgb(x):
            """return sum of three objectives"""
            summe = 0
            quantity = np.array([sum(g) for g in list(grouper(company_count, x))])
            for q in quantity:
                if q <= 10.0:
                    summe += 0.0
                elif q > 10.0 and q <= 50.0:
                    summe += 2.5 * (q-10) 
                else:
                    summe += 100.0

                return summe

        # objective xgboost
        def objective(x, constant_column, kreislauf_column, legierung_column):
            t1 = np.dot(x, price_list)
            list_fremdschrotte = [sum(g) for g in list(grouper(company_count, x))]
            features = np.concatenate((constant_column, kreislauf_column, list_fremdschrotte, legierung_column))
            t2 = f_xgb(features)
            t3 = sum_t3_xgb(x)
            return (t1 + t2 + t3).item()

        # ann prediction 
        @tf.function
        def tf_ann(x,constant_column,kreislauf_column,legierung_column):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            constant_column_tf = tf.convert_to_tensor(constant_column, dtype=tf.float32)
            kreislauf_column_tf = tf.convert_to_tensor(kreislauf_column, dtype=tf.float32)
            legierung_column_tf = tf.convert_to_tensor(legierung_column, dtype=tf.float32)

            list_fremdschrotte = tf.reduce_sum(tf.reshape(x,[-1,company_count]), axis=1)#tf.convert_to_tensor([sum(g) for g in list(grouper(10, x))])
            x = tf.concat([constant_column_tf, kreislauf_column_tf, list_fremdschrotte, legierung_column_tf], axis=0)
            x = tf.convert_to_tensor([x])
            ann_pred = ann_model(x)[0]
            return ann_pred
        
        # function to calculate the logistic cost of tf version
        @tf.function
        def sum_t3_tf(x):
            """
            X: tf.Tensor, the quantity of every scrap
            """
            summe = tf.constant(0.0, dtype=tf.float32)
            quantity = tf.reduce_sum(tf.reshape(x,[-1,company_count]), axis=1) #tf.convert_to_tensor([sum(g) for g in list(grouper(10, x))])

            for q in quantity:
                q = tf.reshape(q, ())
                summe += tf.cond(q <= 10.0, 
                                lambda: 0.0, 
                                lambda: tf.cond(q > 10.0 and q <= 50.0, 
                                                lambda: 2.5 * (q - 10), 
                                                lambda: 100.0))
            return summe

        # function for calculate the total cost, tf version
        @tf.function
        def objective_tf(x,constant_column,kreislauf_column,legierung_column):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            price_list_tf = tf.convert_to_tensor(price_list, dtype=tf.float32)
            t1 = tf.tensordot(x, price_list_tf,axes=1)
            t2 = tf_ann(x,constant_column,kreislauf_column,legierung_column)
            t3 = sum_t3_tf(x)
            
            return t1 + t2 + t3

        # function to calculate the jacobian of tf 
        @tf.function
        def grad_f_ann_tf(x,constant_column,kreislauf_column,legierung_column):
            x1 = tf.convert_to_tensor(x, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(x1)
                t2 = objective_tf(x1,constant_column,kreislauf_column,legierung_column)
                y = tape.jacobian(t2, x1)
                return y
            
        ############################## ML Settings ##############################
        
        ############################## Opt Running ##############################
        
        def optimize_cobyqa(constant_column, kreislauf_column, legierung_column, beq, x_start):
            # Wrap the objective function with a lambda to include the constant variables
            wrapped_objective = lambda x: objective(x, constant_column, kreislauf_column, legierung_column)
            
            start_cobyqa = time.time()
            res_cobyqa = cobyqa.minimize(wrapped_objective, x0=x_start, xl=x_lower, xu=x_upper,
                                        aeq=aeq, beq=beq, options={"disp": False, "maxiter": max_iter})
            end_cobyqa = time.time()

            elapsed_time_cobyqa = end_cobyqa - start_cobyqa
            
            c_violation = (np.dot(aeq, res_cobyqa.x) - beq).tolist()

            return res_cobyqa.x, res_cobyqa.fun, c_violation, elapsed_time_cobyqa
        
        def optimize_pdfo(constant_column, kreislauf_column, legierung_column, beq, x_start):
            wrapped_objective = lambda x: objective(x, constant_column, kreislauf_column, legierung_column)
            
            start_pdfo = time.time()

            def nlc_eq(x):
                return np.dot(aeq, x) - beq

            nonlin_con_eq = {'type': 'eq', 'fun': nlc_eq} 

            res_pdfo = pdfo(wrapped_objective, x_start, bounds=bounds, constraints=[nonlin_con_eq],
                        options={'maxfev': max_iter})

            end_pdfo = time.time()

            elapsed_time_pdfo = end_pdfo - start_pdfo
            
            #c_violation = (np.dot(aeq, res_cobyqa.x) - beq).tolist()
            c_violation = res_pdfo.constr_value[0].tolist()

            return res_pdfo.x, res_pdfo.fun, c_violation, elapsed_time_pdfo, res_pdfo.method
        
        def optimize_grad(constant_column, kreislauf_column, legierung_column, beq, x_start):
            # Wrap the objective and gradient functions with lambda functions
            
            wrapped_objective_tf = lambda x: objective_tf(x.astype(np.float32), constant_column, kreislauf_column, legierung_column)
            wrapped_grad_f_ann_tf = lambda x: grad_f_ann_tf(x.astype(np.float32), constant_column, kreislauf_column, legierung_column)
                
            def equality_fun(x):
                return np.dot(aeq, x) - beq 
            
            eq_cons = {'type': 'eq','fun' : equality_fun}
            
            
            # Initialize the dictionary to store the objective function values
            objective_values = {}

            # Callback function to store the objective function values at each iteration
            def callback(x, n_iter=[1]):
                objective_values[n_iter[0]] = wrapped_objective_tf(x).numpy().item()
                n_iter[0] += 1
            
            start = time.time()
            result = minimize(fun=wrapped_objective_tf, x0=x_start, jac=wrapped_grad_f_ann_tf, constraints=[eq_cons], method='SLSQP',
                            options={'disp': False, 'maxiter':max_iter, 'ftol':1e-9}, bounds=bounds, tol=1e-9,
                            callback=callback)

            
            end = time.time()
            c_violation = (np.dot(aeq, result.x) - beq).tolist()
            elapsed_time = end - start
            return result.x, result.fun, c_violation, elapsed_time, objective_values
        
        ############################## Opt Running ##############################
        opt_result = []
    
        constant_column, kreislauf_column, legierung_column, chemi_to_achieve_fremdschrotte = calculate_chemi_component(df, df_chemi,
                                                                                                                        constant_features_names,kreislauf_schrotte_names,legierung_schrotte_names,fremd_schrotte_names,total_chemi_to_achieve)
        
        beq = chemi_to_achieve_fremdschrotte
        x_start = np.linalg.lstsq(aeq, beq, rcond=None)[0]
        
        
        # check if the optimal schrott list is valid
        fremd_schrotte = df_schrott[df_schrott["name"].str.startswith("F")].copy()
        is_negative = False
        
        while not is_negative:
            print("################# Optimizing for SLSQP iteration #################")

            x_ann, loss_ann, c_violation_ann, elapsed_time_ann, objective_values = optimize_grad(constant_column, kreislauf_column, legierung_column,beq, x_start)
            print("################### original fremd schrotte ###################")
            print(fremd_schrotte["quantity"].to_list())
            # substract the optimal schrott list from the total quantity
            # check ann result if greater than 10 keep it other wise set it to 0
            x_ann = np.where(x_ann > 10, x_ann, 0)
            print("############### ANN result #################", x_ann)
            fremd_schrotte.loc[:, "quantity"] = fremd_schrotte.loc[:,"quantity"].sub(x_ann)
            # check if any value is negative
            is_negative = any(fremd_schrotte["quantity"] < 0)
            print("fremd_schrotte", fremd_schrotte["quantity"].to_list())
            print("------- is negative", is_negative)
            if is_negative:
                # return the message to the frontend
                _data = "The scrap provider does not have enough scrap to provide. Please try again."
                # terminate the optimization process
                services.post(data=_data, target_path=config.TARGET_PATH.get("schrottoptimierung"))
            else:
                # update and save the database of the schrott quantity
                try:
                    Schrott.objects.filter(name__startswith="F").update(quantity=fremd_schrotte["quantity"].to_list())
                    result_current = {}
                    
                    optimal_value = objective(x_ann, constant_column, kreislauf_column, legierung_column)
                    print("objective value", optimal_value)
                    
                    result_current['optimal_value'] = optimal_value
                    result_current['optimal_schrott_list'] = x_ann.tolist()
                    result_current['objective_values'] = objective_values
                    result_current['elapsed_time'] = elapsed_time_ann
                    
                    opt_result.append(result_current)
                    
                    _data = {
                        "optimierung_id": self.optimierung_id,
                        # "optimal_value": opt_result['optimal_value'],
                        # "optimal_schrott_list": opt_result['optimal_schrott_list'],
                        # "objective_values": opt_result['objective_values'],
                        # "elapsed_time": opt_result['elapsed_time'],
                        "opt_result": opt_result,
                    }
                    print("-----------------", _data)
                    
                except Exception as e:
                    _data = "The database is not updated. Please try again."
                    
                finally:
                    services.post(data=_data, target_path=config.TARGET_PATH.get("schrottoptimierung"))
        

class Evaluation:

    def __init__(self, evaluation_id):
        self.threshold_num = 100

        self.evaluation_id = evaluation_id

        params = {"evaluation_id": self.evaluation_id}
        self.evaluation_info = services.get(config.TARGET_PATH.get("evaluation"), params=params)

        # Step 1. Load predictor
        _save_dir = get_experiments_url(self.evaluation_info.get("experiment_id"))
        self.predictor = task.load(_save_dir)

        # Step 2. Load data
        if self.evaluation_info.get("file_id", None) is None:
            # Load test data
            test_url = get_experiments_dataset_url(self.evaluation_info.get("experiment_id"), "test.pickle")

            self.df_data = Input(test_url).from_pickle()

    def evaluate(self):

        self.base_metric()

        if self.predictor.problem_type == "regression":
            self.evaluation_predict_actual()
        else:
            self.evaluation_class()

        self.export_evaluation_result()

    def base_metric(self):
        """
        Calculation basic score for both regression and classification
        :return:
        """
        # Evaluation other scores
        scores = self.predictor.evaluate(self.df_data, model=self.evaluation_info.get("model_name"), silent=True,
                                         detailed_report=True)

        scores.pop("confusion_matrix", None)
        scores.pop("classification_report", None)
        # Post evaluation to server
        _data = {
            "evaluation_id": self.evaluation_id,
            "scores": scores,
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("evaluation"))

    def evaluation_class(self):
        df_y_predict_prob = self.predictor.predict_proba(self.df_data, self.evaluation_info.get("model_name"))
        df_y_actual = self.df_data[self.predictor.label]
        for cur_class in self.predictor.class_labels_internal_map.keys():
            y_actual = df_y_actual.apply(lambda x: 1 if x == cur_class else 0)
            y_predict_prob = df_y_predict_prob[cur_class].values

            class_id = "{}_{}".format(self.evaluation_id, str(cur_class).strip().lower())

            # Post data to save class info
            data = {
                "evaluation_id": self.evaluation_id,
                "class_id": class_id,
                "class_name": str(cur_class).strip().lower()
            }

            services.post(data=data, target_path=config.TARGET_PATH.get("evaluation_class"))

            self.eval_class_roc_lift(y_actual, y_predict_prob, class_id)
            self.eval_class_predict_distri(y_actual, y_predict_prob, class_id)

    def evaluation_predict_actual(self):
        """
        Predict - Actual
        :return:
        """
        if len(self.df_data) > config.MAX_SAMPLE:
            _df_data = self.df_data.sample(n=config.MAX_SAMPLE, replace=True)
        else:
            _df_data = self.df_data

        # Not change location of actual and predict
        predict_vs_actual = {
            #"actual": list(_df_data[self.predictor.label].values),
            "actual": _df_data[self.predictor.label].values.astype(float).tolist(),
            "predict": self.predictor.predict(_df_data, self.evaluation_info.get("model_name")).values.astype(float).tolist()
        }

        data = {
            "evaluation_id": self.evaluation_id,
            "predict_vs_actual": predict_vs_actual,
        }

        services.post(data=data, target_path=config.TARGET_PATH.get("evaluation_predict_actual"))

    def eval_class_roc_lift(self, y_actual, y_predict_prob, class_id):
        scores = []

        base_value = 0

        for threshold in np.linspace(0, 1, self.threshold_num + 1):
            threshold = round(threshold, 3)

            y_predict = np.where(y_predict_prob > threshold, 1, 0)

            tn, fp, fn, tp = confusion_matrix(y_actual, y_predict).ravel()

            recall = recall_score(y_actual, y_predict)

            precision = precision_score(y_actual, y_predict)

            accuracy = accuracy_score(y_actual, y_predict)

            f1 = f1_score(y_actual, y_predict, average="micro")

            # False Positive Rate.
            """ Ti le du doan sai tai not obese class.
            Ex : We have 2 class obese(beo phi) and not obese.
            FPR proportion of not obese samples that were incorrectly classified 
            """
            fpr = fp / (fp + tn)

            # True Positive Rate
            """ Ti le du doan dung tai obese class
            Proportion of obese sample that were correctly classified
            """
            tpr = tp / (tp + fn)

            # Positive predictive value
            ppv = tp / (tp + fp)

            # Base value
            base_value = ppv if threshold == 0 else base_value

            """ Overall population
            Ung voi threshold dang xet, co bao nhieu % data dang duoc su dung.
            Threshold = 0. -> 100% data su dung
            Threshold = 10. -> xx% data duoc su dung. Cach tinh xx = Count( data[threshold > 10] )  
            """
            overall_population = np.where(y_predict_prob >= threshold, 1, 0).sum() / len(self.df_data)

            """Target population
            Ti le du doan dung tren toan bo target
            """
            target_population = tp / y_actual.sum()

            top_percent_of_predict = (tp + fp) / (tp + tn + fp + fn)

            _scores = {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'fpr': fpr, 'tpr': tpr, 'ppv': ppv, 'recall': recall,
                'f1': f1, 'precision': precision,
                'accuracy': accuracy, 'base_value': base_value,
                'threshold': threshold,
                'top_percent_of_predict': top_percent_of_predict,
                'overall_population': overall_population,
                'target_population': target_population,
            }

            _scores = dict([key, parse_float(value)] for key, value in _scores.items())
            scores.append(_scores)

        data = {
            "class_id": class_id,
            "scores": scores
        }

        services.post(data=data, target_path=config.TARGET_PATH.get("evaluation_class_roc_lift"))

    def eval_class_predict_distri(self, y_actual, y_predict_prob, class_id):
        """
        Distribution probability of distribution
        :return:
        """
        df_predict_distribution = pd.DataFrame({"predict_prob":  np.linspace(0, 1, self.threshold_num + 1)})

        df = pd.DataFrame({"predict_prob":  np.round(y_predict_prob, 2), "y_true": y_actual})

        df_cur_class, df_left_class = df.loc[df.y_true == 1], df.loc[df.y_true == 0]

        _target_class = distribution_density(df_cur_class.predict_prob.values, self.threshold_num + 1)
        _left_class = distribution_density(df_left_class.predict_prob.values, self.threshold_num + 1)

        df_predict_distribution["target_class_density"] = _target_class
        df_predict_distribution["left_class_density"] = _left_class

        # Post predict distribution to API Server

        data = {
            "class_id": class_id,
            "predict_distribution": df_predict_distribution.to_dict("list")
        }
        services.post(data=data, target_path=config.TARGET_PATH.get("evaluation_class_distribution"))

    def export_evaluation_result(self):
        _df_predict = self.df_data.copy()

        # Predict
        if self.predictor.problem_type == "regression":
            _predict_result = self.predictor.predict(_df_predict, self.evaluation_info.get("model_name"))

            _df_predict["Predict"] = _predict_result
        else:
            _predict_result = self.predictor.predict_proba(_df_predict, self.evaluation_info.get("model_name"))

            _rename = dict(zip(_predict_result.columns, ["predict_class_{}".format(str(label).strip()) for label
                                                         in _predict_result.columns.tolist()]))

            _predict_result.rename(columns=_rename, inplace=True)

            _df_predict = pd.concat([_df_predict, _predict_result], axis=1)

        # Save file into local
        file_path = get_evaluation_url(self.evaluation_id)
        try:
            excel = Excel(file_path)

            excel.add_data(worksheet_name="Data", pd_data=_df_predict, header_lv0=None, is_fill_color_scale=False,
                           columns_order=None)

            excel.save()
        except Exception as e:
            mes = 'Can not generate excel file.  ERROR : {}'.format(e)
            raise Exception(mes)

    def sub_population(self, sub_population_id, column_name):
        """
        Call this function by run special task. Replace data from DB every time when it called
        :return:
        """

        # TODO : Input which value is positive.
        actual = self.predictor.transform_labels(self.df_data[self.predictor.label].values).values
        predict_raw = self.predictor.predict(self.df_data, self.evaluation_info.get("model_name"))
        predict = self.predictor.transform_labels(predict_raw).values
        feature = self.df_data[column_name].values

        df_sub = pd.DataFrame(
            {
                "feature_name": column_name,
                "feature_value": feature,
                "predict": predict,
                "actual": actual,
                "actual_raw": self.df_data[self.predictor.label].values,
                "predict_raw": predict_raw
            })

        hist = Histogram(df_sub, "feature_value")
        df_allocated_group = hist.pd_data

        if self.predictor.problem_type == "binary":
            df_prob = self.predictor.predict_proba(self.df_data, self.evaluation_info.get("model_name"))
            df_prob.reset_index(drop=True, inplace=True)
            df_allocated_group = pd.concat([df_allocated_group, df_prob], axis=1)

        if self.predictor.problem_type == "regression":
            df_sub_population = self._regression_score(df_allocated_group)
        else:
            # Calculation subpopulation for each class
            df_sub_population = self._binary_score(df_allocated_group, self.predictor.problem_type,
                                                   list(self.predictor.class_labels_internal_map.keys()))

        data = {
            "sub_population_id": sub_population_id,
            "sub_population": df_sub_population.to_dict("records")
        }
        services.post(data, target_path=config.TARGET_PATH.get("evaluation_sub_population"))

    @staticmethod
    def _regression_score(pd_data):
        df_grouped = pd.DataFrame()

        for group in pd_data.group_name.unique():
            df_group = pd_data.loc[pd_data.group_name == group]

            _df_grouped = df_group.groupby(['group_order', 'group_name', 'is_outlier']).agg({"actual": "count"}).rename(
                columns={"actual": "sample"})

            _df_grouped["mean_absolute_error"] = mean_absolute_error(df_group.actual.values, df_group.predict.values)
            _df_grouped["mean_squared_error"] = mean_squared_error(df_group.actual.values, df_group.predict.values)
            _df_grouped["median_absolute_error"] = median_absolute_error(df_group.actual.values,
                                                                         df_group.predict.values)
            _df_grouped["r2"] = r2_score(df_group.actual.values, df_group.predict.values)
            _df_grouped["mean_absolute_percentage_error"] = mean_absolute_percentage_error(df_group.actual.values,
                                                                                           df_group.predict.values)

            df_grouped = pd.concat([df_grouped, _df_grouped])

        df_grouped["sample_percent"] = df_grouped["sample"] / len(pd_data)
        df_grouped = df_grouped.replace([np.nan], [None])

        df_grouped.sort_values("group_order", inplace=True)
        df_grouped.reset_index(inplace=True)

        return df_grouped.reset_index()

    @staticmethod
    def _binary_score(df_data, problem_type, class_list):
        """

        :param pd_data: Allocated group data
        :param problem_type: binary or multiple
        :param class_list: List of target class.
        :return:
        """
        df_grouped = pd.DataFrame()

        """
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        - 'micro':
        Calculate metrics globally by counting the total true positives, false negatives and false positives.
        'binary':
        Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
        """

        if problem_type == "binary":
            f1_score_average = "binary"
        else:
            f1_score_average = "micro"
        # f1_score_average = "binary" if problem_type == "binary" else "micro"

        if df_data.group_name.nunique() > 100:
            group_filter = np.random.choice(df_data.group_name.unique(), 100, replace=False)
            df_data = df_data.loc[df_data.group_name.isin(group_filter)].reset_index(drop=True)

        for group in df_data.group_name.unique():
            df_group = df_data.loc[df_data.group_name == group]

            _df_grouped = df_group.groupby(['group_order', 'group_name', 'is_outlier']).agg({"actual": "count"}).rename(
                columns={"actual": "sample"})

            _df_grouped["accuracy"] = accuracy_score(df_group.actual.values, df_group.predict.values)
            _df_grouped["f1"] = f1_score(df_group.actual.values, df_group.predict.values,
                                         average=f1_score_average)
            _df_grouped["balanced_accuracy_score"] = balanced_accuracy_score(df_group.actual.values,
                                                                             df_group.predict.values)
            _df_grouped["precision_score"] = precision_score(df_group.actual.values, df_group.predict.values,
                                                             average=f1_score_average)
            _df_grouped["recall_score"] = recall_score(df_group.actual.values, df_group.predict.values,
                                                       average=f1_score_average)

            # Calculation confusion matrix
            labels = df_group["actual_raw"].unique().tolist()
            matrix = confusion_matrix(df_group.actual_raw.values, df_group.predict_raw.values, labels).tolist()
            confusion = [{
                "labels": labels,
                "matrix": matrix
            }]
            _df_grouped["confusion"] = confusion

            # TODO: Create distribution chart for binary
            if problem_type == "binary":
                density = []
                for target_class in class_list:
                    cur_class_prob = df_group.loc[df_group.actual_raw == target_class][target_class].values

                    if len(cur_class_prob) == 0:
                        dens, prob = [], []
                    else:
                        dens = distribution_density(cur_class_prob, 30)
                        prob = np.round(np.linspace(0, 1, 30), 2)

                    _dens = {
                        "series": [{"x": x, "y": y} for x, y in zip(prob, dens)],
                        "name": target_class
                    }
                    density.append(_dens)

                _df_grouped["density"] = [density]

            df_grouped = pd.concat([df_grouped, _df_grouped])

        df_grouped["sample_percent"] = df_grouped["sample"] / len(df_data)

        df_grouped = df_grouped.replace([np.nan], [None])

        # Order group by group order
        df_grouped.sort_values("group_order", inplace=True)
        df_grouped.reset_index(inplace=True)

        return df_grouped


class Explain:
    """
    Used SHAP to interpretation models
    """
    def __init__(self, explain_id):

        self.explain_id = explain_id

        # Load explain info
        params = {"explain_id": explain_id}
        self.explain_info = services.get(target_path=config.TARGET_PATH.get("explain"), params=params)

        if self.explain_info.get("file_id", None) is None:
            # Load test data
            test_url = get_experiments_dataset_url(self.explain_info.get("experiment_id"), "test.pickle")

            self.df_data = Input(test_url).from_pickle()

            self.df_data = self.df_data.sample(100)

        # Load predictor
        _save_dir = get_experiments_url(self.explain_info.get("experiment_id"))
        self.predictor = task.load(_save_dir)

        # Calculation shap
        self.shap_values = self.shap_calculation()

    def explain(self):

        self.pdp()

    def pdp(self): 

        pdp_list = []

        for feature in self.predictor.features():

            if self.predictor.problem_type == "regression":
                pdp_values = self.pdp_regress(feature)
            else:
                # Create
                pdp_values = self.pdp_class(feature)

            pdp_list.append(
                {
                    "feature": feature,
                    "pdp_values": pdp_values
                }
            )

        data = {
            "explain_id": self.explain_id,
            "pdp_list": pdp_list
        }
        services.post(data=data, target_path=config.TARGET_PATH.get("explain_pdp"))

    def pdp_regress(self, feature):

        df_shap = self.shap_values

        expected_value = df_shap["expected_value"].unique()[0]

        df_pdp = pd.DataFrame(
            {
                "feature_name": feature,
                "feature_value": self.df_data[feature].values,
                "shap": df_shap[feature].values,
            })

        df_pdp["shap"] = df_pdp["shap"] + expected_value

        hist = Histogram(df_pdp, "feature_value")
        df_pdp_grouped = hist.pd_data.groupby(['group_order', 'group_name']).agg(pdp_value=('shap', "mean"), num=('shap', "count"))
        df_pdp_grouped.reset_index(inplace=True)

        return df_pdp_grouped.to_dict("records")

    def pdp_class(self, feature):

        pdp_class = []

        for class_decode in self.predictor.class_labels_internal_map.keys():

            df_shap = self.shap_values[class_decode]

            expected_value = df_shap["expected_value"].unique()[0]

            df_pdp = pd.DataFrame(
                {
                    "feature_name": feature,
                    "feature_value": self.df_data[feature].values,
                    "shap": df_shap[feature].values,
                })

            df_pdp["shap"] = df_pdp["shap"] + expected_value

            hist = Histogram(df_pdp, "feature_value")

            df_pdp_grouped = hist.pd_data.groupby(['group_order', 'group_name']).agg(pdp_value=('shap', "mean"),num=('shap', "count"))

            df_pdp_grouped.reset_index(inplace=True)

            pdp_class.append({
                "class_name": str(class_decode).strip(),
                "pdp_values": df_pdp_grouped.to_dict("records")
            })

        return pdp_class

    def shap_calculation(self):

        # Calculation shap
        model_name = self.explain_info.get("model_name")
        try:
            model = self.predictor._trainer.load_model(model_name)
        except Exception as e:
            mes = "Can't load model {}. ERROR : {}".format(model_name, e)
            raise mes
        model_type = model.__class__.__name__

        is_tree_explain = model_type in ['RFModel', 'XTModel']

        if is_tree_explain:
            df_data_trans = self.predictor.transform_features(data=self.df_data, model=model_name)

            data = model.preprocess(df_data_trans)
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(data)

        else:
            # Load and Transform train data
            train_url = get_experiments_dataset_url(self.explain_info.get("experiment_id"), "train.pickle")
            df_train = Input(train_url).from_pickle()

            df_train_transform = self.predictor.transform_features(df_train)
            train_summary = shap.kmeans(df_train_transform, 150).data

            # 3. Create model wrapper
            model_wrapper = ModelWrapper(model, self.predictor.features())

            # 4. Shap calculation
            if self.predictor.problem_type != "regression":
                explainer = shap.KernelExplainer(model_wrapper.predict_prob, train_summary)
            else:
                explainer = shap.KernelExplainer(model_wrapper.predict, train_summary)

            # 5. Shap calculation for input data
            df_data_trans = self.predictor.transform_features(data=self.df_data)
            shap_values = explainer.shap_values(df_data_trans.values)

        if self.predictor.problem_type == "regression":
            df_shap = pd.DataFrame(shap_values, columns=model.features)
            df_shap["expected_value"] = explainer.expected_value
            return df_shap
        else:
            shap_values_multi_class = {}
            for class_decode, class_encode in self.predictor.class_labels_internal_map.items():
                df_shap = pd.DataFrame(shap_values[class_encode], columns=model.features)
                df_shap["expected_value"] = explainer.expected_value[class_encode]

                shap_values_multi_class[class_decode] = df_shap

            return shap_values_multi_class

    
######################################## Help function for Schrott Optimizer ########################################

def df_columns_name(df):
    """
    return constant features, schrotte features, kreislauf schrotte, legierung schrotte
    """
    features_columns = df.columns.tolist()
    # remove_columns = ['HeatID', 'HeatOrderID','Energy']
    # features_columns = [x for x in columns if x not in remove_columns]
    
    # constant process parameter, start with "Feature"
    constant_features_names = [x for x in features_columns if "Feature" in x]
    
    #"K" means Kreislauf, "L" means Legierungen, "F" means Fremdschrotte, we only want to optimize "F"
    schrotte_features_names = [x for x in features_columns if "Feature" not in x]
    
    # schrotte name for Kreislauf
    kreislauf_schrotte_names = [x for x in features_columns if "K" in x]
    
    # schrotte name for legierung
    legierung_schrotte_names = [x for x in features_columns if "L" in x]
    
    # schrotte name for Fremdschrotte
    fremd_schrotte_names = [x for x in features_columns if "F" in x and len(x) < 4]
    
    return constant_features_names, schrotte_features_names, kreislauf_schrotte_names,legierung_schrotte_names,fremd_schrotte_names

def calculate_chemi_component(df, df_chemi,
                              constant_features_names,
                              kreislauf_schrotte_names,
                              legierung_schrotte_names,
                              fremd_schrotte_names,
                              total_chemi_to_achieve):
    
    """
    return the randomly chosen row, and its constant column, kreislauf column and 
    legierung column, use them to return the chemical component of fremdschrotte column
    """
    df_random_row = df.sample()
    
    # calculate the constant features
    constant_column = (df_random_row[constant_features_names].values[0]).astype(np.float32)
    
    # calculate the chemical component for kreislauf
    kreislauf_column = (df_random_row[kreislauf_schrotte_names].values[0]).astype(np.float32)
    kreislauf_chemical_table = df_chemi[chemi_names].iloc[len(kreislauf_schrotte_names)-1:]
    chemi_component_kreislauf = (np.dot(kreislauf_column, kreislauf_chemical_table) /100.0).astype(np.float32)
    
    # calculate the chemical component for legierungen
    legierung_column = (df_random_row[legierung_schrotte_names].values[0]).astype(np.float32)
    legierung_chemical_table = df_chemi[chemi_names].iloc[len(fremd_schrotte_names):len(kreislauf_schrotte_names)-1]
    chemi_component_legierung = (np.dot(legierung_column, legierung_chemical_table) /100.0).astype(np.float32)
    
    # calculate the chemical compoent for fremdschrotte
    chemi_to_achieve_fremdschrotte = (np.abs(total_chemi_to_achieve - chemi_component_kreislauf - chemi_component_legierung)).astype(np.float32)
    
    return constant_column, kreislauf_column, legierung_column, chemi_to_achieve_fremdschrotte

def fremdschrott_chemi_table(df_chemi, fremd_schrotte_names,company_count):
    # construct the chemical table
    # assume that every company's chemical elements for every schrott is identical
    df_chemi_fremdschrott= (df_chemi[chemi_names].iloc[:len(fremd_schrotte_names)])
    fremdschrott_chemi = df_chemi_fremdschrott.T 
    n_times = company_count - 1
    temp_dfs = []
    
    for col_name in fremdschrott_chemi.columns:
        temp_df = fremdschrott_chemi[[col_name]].copy()
        for i in range(1, n_times+1):
            temp_df[f'{col_name}{i}'] = fremdschrott_chemi[col_name]
        temp_dfs.append(temp_df)

    fremdschrotte_chemi_table = pd.concat(temp_dfs, axis=1) / 100.0
    
    return fremdschrotte_chemi_table


