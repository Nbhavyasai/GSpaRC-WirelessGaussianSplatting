# #!/bin/bash

# python train.py \
#   --mode eval \
#   --config configs/rfid-spectrum.yml \
#   --dataset_type rfid \
#   --gpu 1 \
#   --load_checkpoint "/home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/results_04_08/run_20260408_140343/checkpoints/checkpoint_iter_200000.pth" \

# python plot_spatial_metrics.py \
#   --rx_csv /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/datasets/sionna/conference-room-mag/rx_pos.csv \
#   --test_csv /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/results_03_18/run_20260318_051923/evaluation/per_sample_metrics_test.csv \
#   --train_csv /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/results_03_18/run_20260318_051923/evaluation/per_sample_metrics_train.csv \
#   --outdir spatial_analysis \

# python local_confidence_from_train_metrics.py \
#   --train_metrics /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/results_03_18/run_20260318_051923/evaluation/per_sample_metrics_train.csv \
#   --train_rx /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/datasets/sionna/conference-room-mag/rx_pos.csv \
#   --test_rx /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/datasets/sionna/conference-room-mag/rx_pos.csv \
#   --test_metrics /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/results_03_18/run_20260318_051923/evaluation/per_sample_metrics_test.csv \
#   --outdir local_confidence \
#   --k 12 \
#   --mae_good_thresh 0.03 \
#   --ssim_good_thresh 0.90 \
#   --use_combined_good_rule \
#   --plots


# python local_confidence_from_nearby_train.py \
#   --rx_csv /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/datasets/sionna/conference-room-mag/rx_pos.csv \
#   --train_csv /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/results_03_18/run_20260318_051923/evaluation/per_sample_metrics_train.csv \
#   --test_csv /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/results_03_18/run_20260318_051923/evaluation/per_sample_metrics_test.csv \
#   --outdir confidence_analysis \
#   --k 8 \
#   --mae_good_thresh 0.02 \
#   --ssim_good_thresh 0.70 \
#   --use_combined_good_rule > confidence_analysis/log.txt

# python analyze_worst_regions.py \
#   --rx_csv ./datasets/sionna/conference-room-mag/rx_pos.csv \
#   --metrics_csv ./results_03_18/run_20260318_051923/evaluation/per_sample_metrics_test.csv \
#   --outdir worst_mae_regions \
#   --metric ssim \
#   --top_percent 5.0 \
#   --cluster_eps 0.7 \
#   --cluster_min_samples 5


# python cluster_test_quality_regions.py \
#   --rx_csv ./datasets/sionna/conference-room-mag/rx_pos.csv \
#   --test_csv ./results_03_18/run_20260318_051923/evaluation/per_sample_metrics_test.csv \
#   --outdir spatial_quality_regions_q \
#   --metric mae \
#   --lower_is_better \
#   --use_quantiles \
#   --good_q 0.33 \
#   --bad_q 0.67

# python spatial_separability_test.py \
#   --rx_csv ./datasets/sionna/conference-room-mag/rx_pos.csv \
#   --test_csv ./results_03_18/run_20260318_051923/evaluation/per_sample_metrics_test.csv \
#   --metric mae \
#   --lower_is_better

# python overlay_test_errors_on_scene.py \
#   --rx_csv ./datasets/sionna/conference-room-mag/rx_pos.csv \
#   --scene_xml ./conference_room/conference_room.xml \
#   --test_csv ./results_03_18/run_20260318_051923/evaluation/per_sample_metrics_test.csv \
#   --metric mae \
#   --show_as_points

# PYVISTA_OFF_SCREEN=true xvfb-run --auto-servernum -s "-screen 0 1280x1024x24" \
# python overlay_test_errors_on_scene.py \
#   --scene_xml ./conference_room/conference_room.xml \
#   --rx_csv ./datasets/sionna/conference-room-mag/rx_pos.csv \
#   --test_csv ./results_03_18/run_20260318_051923/evaluation/per_sample_metrics_test.csv \
#   --metric mae \
#   --sphere_radius 0.06 \
#   --output_html rf_scene_mae.html \
#   --max_meshes 20

# python sionna_overlay.py \
#   --scene_xml /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/conference_room/conference_room.xml \
#   --rx_csv ./datasets/sionna/conference-room-mag/rx_pos.csv \
#   --test_csv ./results_03_18/run_20260318_051923/evaluation/per_sample_metrics_test.csv \
#   --metric mae

python export_gaussians.py \
    --checkpoint /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/results_04_09/run_20260409_212846/checkpoints/checkpoint_iter_300000.pth \
    --output gaussians.npz \
    --opacity_threshold 0.05