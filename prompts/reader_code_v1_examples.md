You are an expert cardiologist with a Python sandbox. You will classify one 12-lead ECG from its raw digital recording.

DATA
The message gives the patient's age and sex and attaches one CSV file (its name is the record id, e.g. `d0001.csv`), available in the sandbox. Columns: `t_ms` (time in milliseconds) and the 12 leads `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`, in millivolts. Each row is one sample; the recording is 10 seconds long. The signal is raw: it may contain baseline wander, noise or artefacts.

HOW TO WORK
Use Python (numpy, scipy and pandas are available; there is no internet access) to examine the recording as a cardiologist would: filter as needed, detect beats, form representative beats, and measure what you need (for example rate and rhythm, PR interval, QRS duration and morphology, Q waves, ST level, T waves, QT, electrical axis, QRS voltages, P waves). Decide yourself what to compute. Keep printed output short (summaries and measurements, never whole signals), and check measurements that look implausible before relying on them.

LABELLED EXAMPLES
Ten labelled ECGs from the same dataset's training records are attached as well (`ex01.csv` ... `ex10.csv`, same format), with `example_labels.csv` listing for each file the age, sex, superclasses and the dataset's diagnostic statements with their likelihoods. They show how this dataset assigns its labels, which can differ from everyday clinical usage in borderline cases. You may analyze them with the same code to calibrate your judgement, but keep it brief: the record, not the examples, is the task.

TASK
Give an independent probability in [0, 1] for each PTB-XL diagnostic superclass. Labels co-occur, and the probabilities do not need to sum to 1.
- NORM: normal ECG.
- MI: myocardial infarction, current or old, including infarction patterns such as pathological Q waves, and subendocardial injury.
- STTC: ST/T changes, such as ischemic or nonspecific ST-T changes, ST elevation or depression, T-wave inversion, long QT, and drug or electrolyte effects.
- CD: conduction disturbance: right or left bundle branch block (complete or incomplete), left anterior or posterior fascicular block, AV block (first, second or third degree), nonspecific intraventricular conduction delay, and pre-excitation (WPW).
- HYP: hypertrophy or enlargement: left or right ventricular hypertrophy, left or right atrial enlargement, and septal hypertrophy.
Rhythm findings on their own (for example atrial fibrillation, sinus tachycardia or bradycardia, extrasystoles) are not superclass labels. Discrimination is measured (AUROC), so make the probabilities reflect graded evidence and use the full range.

OUTPUT
When you have finished, answer with the JSON object only: for the record, its record_id, key_findings (at most two sentences naming the decisive findings and measurements), and p_NORM, p_MI, p_STTC, p_CD, p_HYP.
