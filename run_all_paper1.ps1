Write-Host "======================================================"
Write-Host " Paper 1: AP Method (Calibrated Symmetry Relaxation)"
Write-Host " Full Execution Pipeline (N=7 Seeds)"
Write-Host "======================================================"

cd "d:\downloads\QML Paper ideas\experiments\paper1"

Write-Host "`n---> Running Step 1 (DLA Computation - Seed Independent)"
python step1_dla_computation.py

Write-Host "`n---> Running Step 2 (Gradient Variance at Initialization - 7 Seeds)"
python step2_gradient_variance.py

Write-Host "`n---> Running Step 3 (Commutator Defect - 7 Seeds)"
python step3_commutator_defect.py

Write-Host "`n---> Running Step 4 (VQE Optimization Scale-up - 7/12 Seeds with Checkpointing)"
cd ..\..
python run_phase1_scaleup.py

Write-Host "`n======================================================"
Write-Host " Pipeline Execution Complete."
Write-Host "======================================================"
