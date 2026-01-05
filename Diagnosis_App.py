import joblib
import pickle
import tkinter as tk
import numpy as np

class Diagnosis_App:

    #Initializes syptoms, genders, and a values list.

    def __init__(self):

        self.symptoms = {'Body ache':0,'Cough':1,'Fatigue':2,'Fever':3,'Headache':4,'Runny nose':5,'Shortness of breath':6,'Sore throat':7}
        self.genders = {"Female":0, "Male":1}
        self.values = []

    #Loads the graphics user interface for collecting data.

    def load_gui(self):

        root = tk.Tk()
        root.title("Symptom Checker")
        root.geometry("1000x600")

        age_label = tk.Label(root, text="Age:")
        age_label.grid(row=0, column=0)
        age_entry = tk.Entry(root, width=50)
        age_entry.grid(row=0, column=1)

        gender_label = tk.Label(root, text="Gender:")
        gender_label.grid(row=1, column=0)
        gender_entry = tk.Entry(root, width=50)
        gender_entry.grid(row=1, column=1)

        symptoms_label = tk.Label(root, text="Symptoms List: Body ache, Cough, Fatigue, Fever, Headache, Runny nose, Shortness of breath, Sore throat")
        symptoms_label.grid(row=2, column=0)

        symptom1_label = tk.Label(root, text="Symptom 1:")
        symptom1_label.grid(row=3, column=0)
        symptom1_entry = tk.Entry(root, width=50)
        symptom1_entry.grid(row=3, column=1)      

        symptom2_label = tk.Label(root, text="Symptom 2:")
        symptom2_label.grid(row=4, column=0)
        symptom2_entry = tk.Entry(root, width=50)
        symptom2_entry.grid(row=4, column=1)  

        symptom3_label = tk.Label(root, text="Symptom 3:")
        symptom3_label.grid(row=5, column=0)
        symptom3_entry = tk.Entry(root, width=50)
        symptom3_entry.grid(row=5, column=1)  

        heart_rate_bpm_label = tk.Label(root, text="Heart Rate beats per minute:")
        heart_rate_bpm_label.grid(row=6, column=0)
        heart_rate_bpm_entry = tk.Entry(root, width=50)
        heart_rate_bpm_entry.grid(row=6, column=1) 

        body_temp_label = tk.Label(root, text="Body Temp in Farenheit:")
        body_temp_label.grid(row=7, column=0)
        body_temp_entry = tk.Entry(root, width=50)
        body_temp_entry.grid(row=7, column=1)     

        blood_pressure_mmHg_label = tk.Label(root, text="Blood Pressure in mmHg:")
        blood_pressure_mmHg_label.grid(row=8, column=0)
        blood_pressure_mmHg_entry = tk.Entry(root, width=50)
        blood_pressure_mmHg_entry.grid(row=8, column=1)   

        oxygen_saturation_label = tk.Label(root, text="Oxygen Saturation %:")
        oxygen_saturation_label.grid(row=9, column=0)
        oxygen_saturation_entry = tk.Entry(root, width=50)
        oxygen_saturation_entry.grid(row=9, column=1)   

        return [age_entry, heart_rate_bpm_entry, body_temp_entry, blood_pressure_mmHg_entry, oxygen_saturation_entry, gender_entry, symptom1_entry, symptom2_entry, symptom3_entry, root]

    #Creates button that will submit the data

    def button(self):

        entries = self.load_gui()
        root = entries[-1]
        create_button = tk.Button(root, text="Submit", command=lambda: self.handle_event_data(entries, root))
        create_button.grid(row=10, column=1)
        root.mainloop()
   
    #Extracts the entered value for each field.

    def handle_event_data(self, entries, root):

        for idx, entry in enumerate(entries):

            if idx != len(entries) - 1:
    
                self.values.append(entry.get())

        root.destroy()

    #Farenheit to celisus conversion equation

    def farenheit_to_celisus(self, farenheit):

        return (farenheit - 32) * 5/9.0

    #Turns the blood pressure into a decimal value

    def blood_pressure(self, value):

        values = value.split("/")
        numerator = float(values[0])
        denominator = float(values[1])

        return numerator/denominator

    #Creates array of entered data.

    def vectorize(self):

        zeros = np.zeros((10,1))

        for idx, value in enumerate(self.values):

            if idx == 0 or idx == 1 or idx == 4:

                zeros[idx, 0] = float(value)

            elif idx == 2:

                zeros[idx, 0] = self.farenheit_to_celisus(float(value))
                
            elif idx == 3:

                zeros[idx, 0] = self.blood_pressure(value)

            elif idx == 5:

                if value == "Male":

                    zeros[idx + 1, 0] = 1

                else:

                    zeros[idx, 0] = 1

            elif idx == 6 or idx == 7 or idx == 8:

                if value in self.symptoms.keys():

                    zeros[idx + 1, 0] = self.symptoms[value]


        return zeros.T

    #Loads models

    def load_models(self):

        self.diagnosis_model = joblib.load("diagnosis_model.joblib")
        self.severity_model = joblib.load("severity_model.joblib")
        self.treatment_plan_model = joblib.load("treatment_plan_model.joblib")

    #Loads targets

    def load_targets(self):

        with open("treatment_targets.pkl", "rb") as fh:
            self.treatment_targets = pickle.load(fh)

        with open("diagnosis_targets.pkl", "rb") as fh:
            self.diagnosis_targets = pickle.load(fh)

        self.treatment_targets = list(self.treatment_targets)
        self.diagnosis_targets = list(self.diagnosis_targets)

if __name__ == "__main__":

    diagnosis_app = Diagnosis_App()
    
    action = ""

    while True:

        action = input("Would you like to check your symptoms? Y/n: ")

        if action == "N" or action == "n":

            break

        elif action == "Y" or action == "y":

            #Loading GUI
            diagnosis_app.button()
            #Creating data array
            diagnosis_vector = diagnosis_app.vectorize()
            #Loading targets
            diagnosis_app.load_targets()
            #Loading models
            diagnosis_app.load_models()
            #Generating diagnosis prediction
            diagnosis_prediction = diagnosis_app.diagnosis_model.predict(diagnosis_vector)
            #Updating data array with diagnosis value
            updated_diagnosis_vector = np.concatenate((diagnosis_vector, diagnosis_prediction.reshape(1,1)), axis=1)
            #Generating severity prediction
            severity_prediction = diagnosis_app.severity_model.predict(updated_diagnosis_vector)
            #Updating data array with severity prediction
            updated_diagnosis_vector = np.concatenate((updated_diagnosis_vector, severity_prediction.reshape(1,1)), axis=1)
            #Generating treatment plan prediction
            treatment_plan_prediction = diagnosis_app.treatment_plan_model.predict(updated_diagnosis_vector)
            print("Diagnosis: ", diagnosis_app.diagnosis_targets[diagnosis_prediction[0]])
            print("Treatment Plan: ", diagnosis_app.treatment_targets[treatment_plan_prediction[0]])

        

