import kagglehub
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


class Diagnosis:

    #Initializes data path, loads data into a dataframe and extracts various feature data.

    def __init__(self):
        
        self.path = kagglehub.dataset_download("s3programmer/disease-diagnosis-dataset")
        self.data_path = self.path + "/disease_diagnosis.csv"
        self.df = pd.read_csv(self.data_path)
        self.symptoms = list(set(self.df["Symptom_1"]) | set(self.df["Symptom_2"]) | set(self.df["Symptom_3"]))
        self.severity_levels = list(set(self.df["Severity"]))
        self.diagnosises_set = set(self.df["Diagnosis"])
        self.diagnosises = list(self.diagnosises_set)
        self.features = list(self.df.columns)
        self.features_to_encode = ["Gender"]
        self.target_features = ["Treatment_Plan", "Symptom_1", "Symptom_2", "Symptom_3", "Diagnosis", "Severity"]
        self.features_to_drop = self.features_to_encode + self.target_features + ['Patient_ID']
        self.y_target_features = ["new_Diagnosis", "new_Severity", "new_Treatment_Plan"]
        self.treatment_plan_set = set(self.df["Treatment_Plan"])

    #One hot encodes features

    def feature_encoder(self, dataframe, feature, columns):

        values = dataframe[feature].values
        element_list = list(set(values))
        element_array = np.array(element_list)
        zeros = np.zeros((len(values), len(element_array)))

        for row_idx, value in enumerate(values):

            col_idx = np.where(element_array == value)
            zeros[row_idx, col_idx] = 1

        updated_columns = columns + element_list

        updated_values = np.concatenate((dataframe.values, zeros), axis=1)
        updated_df = pd.DataFrame(updated_values,columns=updated_columns)

        return updated_df, updated_columns

    #Encodes features with multiclass labels

    def target_encoder(self, dataframe, feature, targets=None):

        values = dataframe[feature].values
        element_list = list(set(values))
        element_array = np.array(element_list)
        zeros = np.zeros((len(values),1))

        for row_idx, value in enumerate(values):

            encoding = np.where(element_array == value)[0]
            zeros[row_idx][0] = encoding

        columns = list(dataframe.columns) + ["new_" + feature]
        updated_values = np.concatenate((dataframe.values, zeros), axis=1)
        updated_df = pd.DataFrame(updated_values,columns=columns)

        return updated_df, element_array

    #Creates a decimal value from dividing the blood pressure

    def convert_blood_pressure(self, dataframe, feature):

        values = dataframe[feature].values
        zeros = np.zeros((len(values), 1))

        for idx, value in enumerate(values):

            blood_pressure_values = value.split("/")
            numerator = int(blood_pressure_values[0])
            denominator = int(blood_pressure_values[1])
            zeros[idx][0] = numerator/denominator

        dataframe[feature] = zeros

        return dataframe

    #Calls the feature encoder on all the features needing encoding

    def update_df_features(self):

        for idx, feature in enumerate(self.features_to_encode):
            if idx == 0:
                self.updated_df, self.updated_columns = self.feature_encoder(self.df, feature, self.features)
            else:
                self.updated_df, self.updated_columns = self.feature_encoder(self.updated_df, feature, self.updated_columns)

    #Callsthe target encoder on all targets needing encoding.

    def update_target_features(self):

        for idx, feature in enumerate(self.target_features):
            if idx == 0:
                self.target_encoded_df, self.targets = self.target_encoder(self.updated_df, feature)
            else:
                self.target_encoded_df, _ = self.target_encoder(self.target_encoded_df, feature)

    #Processes all dataframes and returns final dataframe

    def finalize_df(self):
        
        self.update_df_features()
        self.update_target_features()
        self.blood_pressure_converted_df = self.convert_blood_pressure(self.target_encoded_df, "Blood_Pressure_mmHg")
        self.final_cols = [column for column in self.blood_pressure_converted_df.columns if column not in self.features_to_drop]
        self.final_df = self.target_encoded_df[self.final_cols]

    #Extracts data and target values for three separate targets sets and divides them into training and test sets.

    def processing_df_with_all_features(self):

        x_cols = [col for col in self.final_df.columns if col not in self.y_target_features]

        y_treatment_plan_idx = np.where(np.array(self.final_df.columns) == "new_Treatment_Plan")
        y_treatment_plan_cols = list(np.array(self.final_df.columns)[y_treatment_plan_idx])

        y_diagnosis_idx = np.where(np.array(self.final_df.columns) == "new_Diagnosis")
        y_diagnosis_cols = list(np.array(self.final_df.columns)[y_diagnosis_idx])

        y_severity_idx = np.where(np.array(self.final_df.columns) == "new_Severity")
        y_severity_cols = list(np.array(self.final_df.columns)[y_severity_idx])

        x = self.final_df[x_cols].values
        y_treatment_plan = self.final_df[y_treatment_plan_cols].values.reshape(len(self.final_df,))
        y_diagnosis = self.final_df[y_diagnosis_cols].values.reshape(len(self.final_df,))
        y_severity = self.final_df[y_severity_cols].values.reshape(len(self.final_df,))
        
        self.n_idx = int(len(x) * .8)
        
        self.x_train, self.x_test = x[:self.n_idx,:], x[self.n_idx:,:]
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)

        self.y_train_treatment, self.y_test_treatment = y_treatment_plan[:self.n_idx], y_treatment_plan[self.n_idx:]
        self.y_train_treatment, self.y_test_treatment = self.y_train_treatment.astype(np.int32), self.y_test_treatment.astype(np.int32)
        
        self.y_train_diagnosis, self.y_test_diagnosis = y_diagnosis[:self.n_idx], y_diagnosis[self.n_idx:]
        self.y_train_diagnosis, self.y_test_diagnosis = self.y_train_diagnosis.astype(np.int32), self.y_test_diagnosis.astype(np.int32)  

        self.y_train_severity, self.y_test_severity = y_severity[:self.n_idx], y_severity[self.n_idx:]
        self.y_train_severity, self.y_test_severity = self.y_train_severity.astype(np.int32), self.y_test_severity.astype(np.int32)            

        self.treatment_targets = np.array(list(set(list(self.y_train_treatment)) | set(list(self.y_test_treatment))))
        self.diagnosis_targets = np.array(list(set(list(self.y_train_diagnosis)) | set(list(self.y_test_diagnosis))))
        self.severity_targets = np.array(list(set(list(self.y_train_severity)) | set(list(self.y_test_severity))))

        self.save_targets(self.diagnosises_set, "diagnosis_targets.pkl")
        self.save_targets(self.treatment_plan_set, "treatment_targets.pkl")

        self.target_dictionary = {"new_Diagnosis":self.diagnosis_targets, "new_Severity":self.severity_targets, "new_Treatment_Plan":self.severity_targets}

    #Saves target values

    def save_targets(self, targets, filename):

        with open(filename, "wb") as fh:
            pickle.dump(targets, fh)

    #Instantiates random forest model and fits the training data while making predictions.

    def model(self, x_train, y_train, x_test, y_test, targets, feature):

        self.current_model = RandomForestClassifier(max_depth=10, random_state=42)
        self.current_model.fit(x_train, y_train)

        self.final_train_predictions = self.current_model.predict(x_train)
        self.final_test_predictions = self.current_model.predict(x_test)

        if feature == "new_Diagnosis":
            print(feature)
            print("Test accuracy: ", self.accuracy(self.final_test_predictions, self.y_test_diagnosis))
        elif feature == "new_Serverity":
            print(feature)
            print("Test accuracy: ", self.accuracy(self.final_test_predictions, self.y_test_severity))
        else:
            print(feature)
            print("Test accuracy: ", self.accuracy(self.final_test_predictions, self.y_test_treatment))

    #Adds data column to array

    def add_col(self, x, col):

        col_shape = col.reshape((len(col), 1))
        
        return np.concatenate((x, col_shape), axis=1)

    #Saves model

    def save_model(self, model, filename):

        joblib.dump(model, filename)

    #Runs a random forest model to find a diagnosis then updates the data array with the diagnosis and runs a random forest to determine the severity then
    #adds that sample to the data array and finally uses the random forest model to determine a treatment plan.

    def models(self, prediction_features):

        for idx, feature in enumerate(prediction_features):
            if idx == 0:
                self.model(self.x_train, self.y_train_diagnosis, self.x_test, self.y_test_diagnosis, self.target_dictionary[feature], feature)
                self.train_col_to_add = self.final_train_predictions.reshape(len(self.final_train_predictions), 1)
                self.updated_x_train = self.add_col(self.x_train, self.train_col_to_add)
                self.test_col_to_add = self.final_test_predictions.reshape(len(self.final_test_predictions), 1)
                self.updated_x_test = self.add_col(self.x_test, self.test_col_to_add)
                self.save_model(self.current_model, "diagnosis_model.joblib")
            elif (idx != 0 and idx != len(prediction_features) - 1):
                self.model(self.updated_x_train, self.y_train_severity, self.updated_x_test, self.y_test_severity, self.target_dictionary[feature], feature)
                self.train_col_to_add = self.final_train_predictions.reshape(len(self.final_train_predictions), 1)
                self.updated_x_train = self.add_col(self.updated_x_train, self.train_col_to_add)
                self.test_col_to_add = self.final_test_predictions.reshape(len(self.final_test_predictions), 1)
                self.updated_x_test = self.add_col(self.updated_x_test, self.test_col_to_add)
                self.save_model(self.current_model, "severity_model.joblib")
            else:
                self.model(self.updated_x_train, self.y_train_treatment, self.updated_x_test, self.y_test_treatment, self.target_dictionary[feature], feature)
                self.save_model(self.current_model, "treatment_plan_model.joblib")

    #Determines the accuracy of the model.

    def accuracy(self, predictions, targets):

        return (np.sum(((predictions == targets) * 1) / len(targets)) * 100)

if __name__ == "__main__":

    diagnosis = Diagnosis()
    diagnosis.finalize_df()
    diagnosis.processing_df_with_all_features()
    diagnosis.models(prediction_features=diagnosis.y_target_features)

        