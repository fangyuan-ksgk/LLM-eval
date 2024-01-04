import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arthur_bench.run.testsuite import TestSuite


eli5 = pd.read_csv('./specificity/eli5_25.csv')
eli5.head(1)

# TestSuite contains scorer & question & context
suite_spec = TestSuite(
    name='specificity',
    scoring_method='specificity',
    reference_data=eli5,
    input_column='history'
)

# run test 
# Question: What are we testing here? The question, the candidate answers and the ratings are all in the dataframe. 
# Since we are not trying to train anything here, we must be aiming for something missing from the dataframe?
# Or are we just trying to mask the score, and compare the predicted score with the actual score? -- that is possible
run_A = suite_spec.run(
    run_name="A",
    candidate_data=eli5,
    candidate_column='human_ref_A'
)

run_B = suite_spec.run(
    run_name="B",
    candidate_data=eli5,
    candidate_column='human_ref_B'
)

# It looks like there is only one test case here.
A_scores = []
for t in run_A.test_cases:
    A_scores.append(t.score)

B_scores = []
for t in run_B.test_cases:
    B_scores.append(t.score)

print(len(A_scores), len(B_scores))