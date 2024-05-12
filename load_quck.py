import fiftyone as fo
import fiftyone.zoo as foz
import json
import os

# Load the JSON file
dataset_path = '/path/to/samples.json'  # Update this path to where you've stored 'samples.json'
with open(dataset_path, 'r') as f:
    data = json.load(f)
    samples = data['samples']

# Create a new FiftyOne dataset
dataset = fo.Dataset('quickstart_manual')

# Process and add each sample to the dataset
for sample_data in samples:
    # Create a FiftyOne sample
    sample = fo.Sample(filepath=os.path.join('/absolute/path/to/data', sample_data['filepath']))  # Update path as necessary

    # Add metadata if available
    if sample_data['metadata']:
        sample.metadata = fo.ImageMetadata(**sample_data['metadata'])

    # Add uniqueness score
    sample['uniqueness'] = sample_data['uniqueness']

    # Add ground truth detections
    if 'ground_truth' in sample_data and sample_data['ground_truth']['detections']:
        detections = []
        for det in sample_data['ground_truth']['detections']:
            detection = fo.Detection(
                label=det['label'],
                bounding_box=det['bounding_box'],
                confidence=det.get('confidence'),  # Include confidence if available
                attributes={'area': fo.NumericAttribute(value=det['area'])} if 'area' in det else None
            )
            detections.append(detection)
        sample['ground_truth'] = fo.Detections(detections=detections)

    # Add predictions if available
    if 'predictions' in sample_data and sample_data['predictions']['detections']:
        predictions = []
        for pred in sample_data['predictions']['detections']:
            prediction = fo.Detection(
                label=pred['label'],
                bounding_box=pred['bounding_box'],
                confidence=pred.get('confidence'),  # Include confidence if necessary
            )
            predictions.append(prediction)
        sample['predictions'] = fo.Detections(detections=predictions)

    # Add tags if available
    if 'tags' in sample_data:
        sample.tags = sample_data['tags']

    # Save the sample
    dataset.add_sample(sample)

# Launch the FiftyOne app
session = fo.launch_app(dataset)
