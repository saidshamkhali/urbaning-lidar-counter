# Method

This document explains every stage of the pipeline and the reasoning behind it.
Code references point at the modules in [`ulc/`](../ulc).

## 0. Problem setting

* **Input:** four pole-mounted LiDARs at `crossing1` of UrbanIng-V2X (two poles, each carrying one
  upright and one upside-down sensor ~5 m above the road). They are time-synchronised and calibrated
  to one global frame. There are ~120 k points per fused frame, 10 Hz, 200 frames (20 s) per sequence.
* **Output:** for every frame, oriented 3D boxes `[x, y, z, l, w, h, yaw]` with class, score and
  track ID, the number of vehicles in the scene, and the number of unique vehicles seen in the
  sequence.
* **Constraint:** only infrastructure LiDARs, no cameras and no vehicle sensors.

The sensors never move. That single fact drives the design: we can learn what the *empty*
intersection looks like and focus on what is different, and the time dimension is ours to use.

## 1. Fusion ([`ulc/io.py`](../ulc/io.py))

Each sweep is transformed with its `gTl` extrinsic into the global ENU frame (origin = GPS origin of
crossing1) and the four sweeps listed for a time step in `timesync_info.csv` are concatenated.
The dataset's Lanelet2 HD map is projected into the same frame (UTM zone 32 relative to the crossing
origin, the same convention as the devkit).

## 2. Static scene model ([`ulc/background.py`](../ulc/background.py))

* **Ground.** The devkit's quadratic ground surface is refined with a 0.5 m elevation map: the 15th
  percentile of near-ground heights per cell over all frames, hole-filled and median-filtered. This
  absorbs curbs, sidewalks and road camber.
* **Occupancy statistics.** For every 0.25 m voxel (0.25–4.5 m above ground) we measure the fraction
  of frames in which it is occupied, separately for each sequence. The three sequences were recorded
  on two different days, so:
  * voxels occupied in *every* sequence are **permanent structure** (façades, poles, trees, signs).
    They are removed before clustering. We checked that this removes none of the labelled visible vehicles.
  * voxels occupied in most frames of *one* sequence are typically **parked vehicles**. They are kept
    and exposed as a feature.
  * the rest is **transient foreground** (moving traffic).

## 3. Clustering and box fitting ([`ulc/detect.py`](../ulc/detect.py))

1. Object-candidate points are clustered in bird's-eye view with DBSCAN (ε = 0.55 m). Clusters longer
   than a bus are re-split with a tighter ε.
2. **L-shape fitting** (Zhang et al., 2017) finds the heading that best explains the points as two
   perpendicular visible sides (closeness criterion, 1° search).
3. **HD-map lane prior.** Half of all vehicle clusters have fewer than 30 points, because they are far
   away or partly occluded, and on such partial views the L-shape heading is basically random: 83 %
   were more than 30° off. We therefore build a *lane direction field* from the map's lane markings,
   curbs and road borders. Segment directions are accumulated as doubled-angle vectors so that
   opposite directions agree, then blurred, and the normalised vector length gives a coherence. For
   sparse clusters on a coherent road arm the heading is snapped to the lane. Inside the junction
   (< 18 m from the centre), where lanes cross, the heading is left to the tracker's motion estimate.
   Effect: median IoU of sparse car clusters went from 0.21 to 0.53.
4. **Amodal completion.** A LiDAR only sees the faces turned towards it, so the tight box is grown
   to a class-specific size prior *away from the nearest sensor*.
5. **Fragment merging.** Vehicle detections whose boxes overlap, or whose point sets are < 1 m apart,
   are merged if the union still has a plausible vehicle size.

## 4. Learned cluster classifier ([`ulc/classify.py`](../ulc/classify.py))

Every cluster is described by 25 features:

* geometry: observed length, width, height percentiles, aspect, area, PCA linearity, planarity and scatter
* sampling: point count, density, and point count normalised by range²
* reflectivity: intensity statistics
* scene context: fraction of transient points, mean occupancy frequency, and whether the cluster lies on
  a road, walkway or vegetation area of the HD map

**Training labels are mined automatically.** A cluster gets the class of the ground-truth box that
contains ≥ 50 % of its points, or `background`. A gradient-boosted tree ensemble
(`HistGradientBoostingClassifier`, class-balanced) is trained with **leave-one-sequence-out**, so
every sequence is always processed by a model that has never seen it.

## 5. Tracking ([`ulc/tracking.py`](../ulc/tracking.py))

A multi-object tracker with a constant-velocity Kalman filter per track:

* **Association:** Hungarian matching with class-aware gates.
* **Heading:** resolved from the motion direction for moving objects.
* **Size and class:** robust size smoothing and score-weighted class voting.
* **Track life-cycle:** coasting through short occlusions.

Because we analyse recorded sequences, an offline pass also:

* removes short, low-confidence tracks;
* back-fills the tentative period of confirmed tracks;
* merges fragments of the same vehicle;
* smooths trajectories.

## 6. Counting

* **Per frame:** number of confirmed vehicle tracks present (car, van, truck, bus), split into moving
  (> 1 m/s) and parked.
* **Per sequence:** number of unique vehicle tracks.

## 7. Evaluation ([`ulc/evaluate.py`](../ulc/evaluate.py))

The UrbanIng-V2X labels were created with vehicle sensors as well, and many labelled cars are
**never seen by the infrastructure LiDARs** (e.g. a parking lot behind a building): only ~55–65 % of
labelled vehicles contain ≥ 5 infrastructure points in a given frame. Evaluating against all labels
would mostly measure sensor coverage, so we report three GT references:

| reference | definition | used for |
|---|---|---|
| **visible** | ≥ 5 infra-LiDAR points inside the box in this frame | detection AP / precision / recall (others are KITTI-style *don't care*) |
| **trackable** (in coverage) | vehicle visible in ≥ 5 frames of the sequence, counted in every frame it exists | counting (a tracker *should* remember a car hidden behind a bus) |
| **all labels** | everything in the label file | shown for context |

Metrics:

* **Detection:** AP at BEV IoU 0.3 / 0.5 (vehicle group, class-agnostic), precision/recall/F1, and recall by distance.
* **Counting:** MAE, RMSE and bias of the per-frame vehicle count, plus the unique-count error.
* **Tracking:** CLEAR-MOT (MOTA, MOTP) and ID switches.

The classifier is evaluated strictly leave-one-sequence-out. Some hyper-parameters of the geometric
stages and the tracker were chosen by looking at these same three sequences (there are no others for
this crossing in the task), so the numbers should be read as *development-set* results.
