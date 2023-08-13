import pdfplumber
import re
import os
import argparse
import torch
import shutil
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CVAnalyzer:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.classes = [
    'ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE', 'APPAREL', 'ARTS', 'AUTOMOBILE',
    'AVIATION', 'BANKING', 'BPO', 'BUSINESS-DEVELOPMENT', 'CHEF', 'CONSTRUCTION',
    'CONSULTANT', 'DESIGNER', 'DIGITAL-MEDIA', 'ENGINEERING', 'FINANCE', 'FITNESS',
    'HEALTHCARE', 'HR', 'INFORMATION-TECHNOLOGY', 'PUBLIC-RELATIONS', 'SALES', 'TEACHER'
]
        
    def extract_text_from_pdf(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            cv_text = ""
            for page in pdf.pages:
                cv_text += page.extract_text()
        return cv_text
    
    def analyze_cv(self, cv_text):
        # Your analysis logic here
        predicted_class_index = 0  # Replace with actual classification result
        predicted_class = self.classes[predicted_class_index]
        return predicted_class
    
    def process_cvs(self, cv_folder):
        cv_list = os.listdir(cv_folder)
        result_data = []
        
        for cv_file in cv_list:
            cv_name = os.path.splitext(cv_file)[0]
            cv_path = os.path.join(cv_folder, cv_file)
            
            cv_text = self.extract_text_from_pdf(cv_path)
            predicted_category = self.analyze_cv(cv_text)
            
            result_data.append([cv_name, predicted_category, cv_path])
        
        return result_data
    
    def move_cvs_to_folders(self, result_data, output_folder):
        for cv_name, predicted_category, cv_path in result_data:
            category_folder = os.path.join(output_folder, predicted_category)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            target_cv_path = os.path.join(category_folder, os.path.basename(cv_path))
            shutil.move(cv_path, target_cv_path)
    
    def dump_results_to_csv(self, result_data, output_csv):
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["cv-name", "predicted-category", "path"])
            writer.writerows(result_data)

def main():
    parser = argparse.ArgumentParser(description="Predict categories from a folder of CVs")
    parser.add_argument("--cv-folder", type=str, required=True, help="Path to the folder containing CVs")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model")
    args = parser.parse_args()
    
    cv_analyzer = CVAnalyzer(args.model_path)
    result_data = cv_analyzer.process_cvs(args.cv_folder)
    
    output_folder = "output_categories"
    output_csv = os.path.join(output_folder, "cv_predictions.csv")
    
    cv_analyzer.move_cvs_to_folders(result_data, output_folder)
    cv_analyzer.dump_results_to_csv(result_data, output_csv)
    
    print("CV analysis and categorization complete.")

if __name__ == "__main__":
    main()