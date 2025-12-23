import unittest
import joblib
import sys
import os

# main.py dosyasındaki fonksiyonu çağırıyoruz
try:
    from main import clean_python_code 
except ImportError:
    print("main.py bulunamadı.")
    sys.exit(1)

# Test edilecek dosyalar
TEST_VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
TEST_MODEL_PATH = 'model_logistic_regression.pkl'

class TestModelFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vectorizer = None
        cls.model = None
        try:
            if os.path.exists(TEST_VECTORIZER_PATH) and os.path.exists(TEST_MODEL_PATH):
                cls.vectorizer = joblib.load(TEST_VECTORIZER_PATH)
                cls.model = joblib.load(TEST_MODEL_PATH)
        except Exception as e:
            print(f"Hata: {e}")
        
    # Test Case 1
    def test_1_code_cleaning(self):
        input_code = """
# Yorum
def test():
    pass
        """
        expected_output = "def test():\n    pass"
        cleaned_code = clean_python_code(input_code)
        self.assertEqual(cleaned_code, expected_output)

    # Test Case 2
    def test_2_model_prediction_output_format(self):
        if self.model is None:
            self.skipTest("Model yüklenemedi.")
            return

        test_text = "print('hello')"
        vectorized_data = self.vectorizer.transform([test_text])
        probabilities = self.model.predict_proba(vectorized_data)[0]
        
        self.assertEqual(len(probabilities), 2)
        self.assertTrue(0 <= probabilities[0] <= 1)
        self.assertTrue(0 <= probabilities[1] <= 1)
        
    # Test Case 3
    def test_3_critical_files_existence(self):
        self.assertTrue(self.vectorizer is not None)
        self.assertTrue(self.model is not None)
        self.assertTrue(hasattr(self.vectorizer, 'transform'))
        self.assertTrue(hasattr(self.model, 'predict_proba'))

if __name__ == '__main__':
    unittest.main()