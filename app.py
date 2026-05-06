"""
Deep Learning Toolbox - Flask Application
Main entry point for the application
"""

import os

from flask import Flask, render_template, request, jsonify
from perceptron import BUILTIN_DATASETS, get_dataset_csv, predict_from_model, train_from_csv
from backpropogation import (
    BUILTIN_DATASETS as BACKPROP_BUILTIN_DATASETS,
    get_dataset_csv as get_backprop_dataset_csv,
    predict_from_model as predict_backprop_model,
    train_from_csv as train_backprop_from_csv,
)
from mlp import (
    BUILTIN_DATASETS as MLP_BUILTIN_DATASETS,
    get_dataset_csv as get_mlp_dataset_csv,
    list_builtin_datasets as list_mlp_builtin_datasets,
    train_from_csv as train_mlp_from_csv,
)
from activation_functions import (
    analyze_activation,
    list_activation_cards,
    predict_uploaded_image,
    project_config,
    train_project,
)
from cnn import (
    perform_convolution_demo,
    perform_pooling_demo,
    perform_activation_demo,
    perform_image_convolution_lab,
)
from opencv import detect_objects_from_base64
from rnn import train_rnn_model, get_rnn_architecture_info
from rnn_project.sentiment_engine import (
    train_sentiment_model,
    predict_sentiment,
    train_notebook_style_model,
    predict_sentiment_notebook_style,
    save_notebook_style_model_to_pkl,
    load_notebook_style_model_from_pkl,
)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'

# Routes
@app.route('/')
def index():
    """Landing page / Dashboard"""
    return render_template('index.html')


@app.route('/perceptron')
def perceptron():
    """Perceptron learning page - dashboard with trainer modal"""
    return render_template('perceptron_dashboard.html')


@app.route('/backpropogation')
def backpropogation():
    """Backpropagation learning page"""
    return render_template('backpropogation_dashboard.html')


@app.route('/mlp')
def mlp():
    """Multi-layer perceptron page"""
    return render_template('mlp_dashboard.html')


@app.route('/activation_functions')
def activation_functions_dashboard():
    """Activation functions learning page"""
    return render_template('activation_functions_dashboard.html')


@app.route('/digit_detection')
def digit_detection_dashboard():
    """MNIST single-neuron project dashboard"""
    return render_template('digit_detection_dashboard.html')


@app.route('/cnn')
def cnn_dashboard():
    """Convolutional Neural Networks learning page"""
    return render_template('cnn_dashboard.html')


@app.route('/opencv_project')
def opencv_project_dashboard():
    """YOLO OpenCV object/person detection project page."""
    return render_template('opencv_dashboard.html')


@app.route('/rnn_project')
def rnn_project_dashboard():
    """RNN project page for learning recurrent neural networks."""
    return render_template('rnn_dashboard.html')


@app.route('/sentiment_analysis')
def sentiment_analysis_dashboard():
    """Sentiment analysis learning page - LSTM-based classification."""
    return render_template('sentiment_analysis_dashboard.html')


@app.route('/about')
def about():
    """About page - placeholder"""
    return {'message': 'About page - coming soon'}


@app.route('/api/perceptron/dataset/<gate_name>', methods=['GET'])
def perceptron_dataset(gate_name):
    """Return built-in CSV dataset for a logic gate."""
    try:
        csv_data = get_dataset_csv(gate_name)
        return jsonify({'success': True, 'gate': gate_name.lower(), 'dataset_csv': csv_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/perceptron/train', methods=['POST'])
def perceptron_train():
    """Train perceptron from CSV using notebook-style update logic."""
    try:
        data = request.json or {}

        gate_name = str(data.get('gate', '')).lower().strip()
        dataset_csv = data.get('dataset_csv')

        if (not dataset_csv) and gate_name in BUILTIN_DATASETS:
            dataset_csv = get_dataset_csv(gate_name)

        if not dataset_csv:
            return jsonify({'error': 'dataset_csv is required for custom training.'}), 400

        learning_rate = float(data.get('learning_rate', 0.1))
        epochs = int(data.get('epochs', 50))
        initial_bias = float(data.get('initial_bias', 0.0))
        initial_weights = data.get('initial_weights', [])

        result = train_from_csv(
            dataset_csv=str(dataset_csv),
            learning_rate=learning_rate,
            epochs=epochs,
            initial_weights=initial_weights,
            initial_bias=initial_bias,
        )

        return jsonify({'success': True, 'gate': gate_name or 'custom', **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/perceptron/predict', methods=['POST'])
def perceptron_predict():
    """Predict using already-trained perceptron weights and bias."""
    try:
        data = request.json or {}
        weights = [float(v) for v in data.get('weights', [])]
        bias = float(data.get('bias', 0.0))
        inputs = [float(v) for v in data.get('inputs', [])]

        if not weights:
            return jsonify({'error': 'weights are required.'}), 400

        result = predict_from_model(weights=weights, bias=bias, inputs=inputs)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/backprop/dataset/<dataset_name>', methods=['GET'])
def backprop_dataset(dataset_name):
    """Return built-in CSV dataset for backpropagation demos."""
    try:
        csv_data = get_backprop_dataset_csv(dataset_name)
        return jsonify({'success': True, 'dataset': dataset_name.lower(), 'dataset_csv': csv_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/backprop/train', methods=['POST'])
def backprop_train():
    """Train tiny backpropagation network from CSV."""
    try:
        data = request.json or {}

        dataset_name = str(data.get('dataset', '')).lower().strip()
        dataset_csv = data.get('dataset_csv')

        if (not dataset_csv) and dataset_name in BACKPROP_BUILTIN_DATASETS:
            dataset_csv = get_backprop_dataset_csv(dataset_name)

        if not dataset_csv:
            return jsonify({'error': 'dataset_csv is required for custom training.'}), 400

        learning_rate = float(data.get('learning_rate', 0.5))
        epochs = int(data.get('epochs', 10))
        hidden_neurons = int(data.get('hidden_neurons', 2))

        result = train_backprop_from_csv(
            dataset_csv=str(dataset_csv),
            learning_rate=learning_rate,
            epochs=epochs,
            hidden_neurons=hidden_neurons,
            initial_input_hidden=data.get('initial_input_hidden'),
            initial_hidden_output=data.get('initial_hidden_output'),
            initial_hidden_bias=data.get('initial_hidden_bias'),
            initial_output_bias=data.get('initial_output_bias'),
        )

        return jsonify({'success': True, 'dataset': dataset_name or 'custom', **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/backprop/predict', methods=['POST'])
def backprop_predict():
    """Predict with trained backprop model parameters."""
    try:
        data = request.json or {}

        weights_input_hidden = data.get('weights_input_hidden', [])
        weights_hidden_output = [float(v) for v in data.get('weights_hidden_output', [])]
        bias_hidden = [float(v) for v in data.get('bias_hidden', [])]
        bias_output = float(data.get('bias_output', 0.0))
        inputs = [float(v) for v in data.get('inputs', [])]

        if not weights_input_hidden:
            return jsonify({'error': 'weights_input_hidden are required.'}), 400

        normalized_input_hidden = []
        for row in weights_input_hidden:
            normalized_input_hidden.append([float(v) for v in row])

        result = predict_backprop_model(
            weights_input_hidden=normalized_input_hidden,
            weights_hidden_output=weights_hidden_output,
            bias_hidden=bias_hidden,
            bias_output=bias_output,
            inputs=inputs,
        )
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/mlp/datasets', methods=['GET'])
def mlp_datasets():
    """Return available MLP built-in datasets."""
    try:
        return jsonify({'success': True, 'datasets': list_mlp_builtin_datasets()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/mlp/dataset/<dataset_name>', methods=['GET'])
def mlp_dataset(dataset_name):
    """Return built-in CSV dataset for MLP trainer."""
    try:
        csv_data = get_mlp_dataset_csv(dataset_name)
        return jsonify({'success': True, 'dataset': dataset_name.lower(), 'dataset_csv': csv_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/mlp/train', methods=['POST'])
def mlp_train():
    """Train manual MLP from CSV with notebook-like preprocessing."""
    try:
        data = request.json or {}

        dataset_name = str(data.get('dataset', '')).lower().strip()
        dataset_csv = data.get('dataset_csv')

        if (not dataset_csv) and dataset_name in MLP_BUILTIN_DATASETS:
            dataset_csv = get_mlp_dataset_csv(dataset_name)

        if not dataset_csv:
            return jsonify({'error': 'dataset_csv is required for custom training.'}), 400

        target_column = data.get('target_column')
        selected_feature_columns = data.get('selected_feature_columns')

        result = train_mlp_from_csv(
            dataset_csv=str(dataset_csv),
            target_column=target_column,
            selected_feature_columns=selected_feature_columns,
            row_threshold_for_drop=int(data.get('row_threshold_for_drop', 1000)),
            test_size=float(data.get('test_size', 0.2)),
            random_state=int(data.get('random_state', 42)),
            learning_rate=float(data.get('learning_rate', 0.01)),
            epochs=int(data.get('epochs', 250)),
            batch_size=int(data.get('batch_size', 32)),
            hidden_layers_override=data.get('hidden_layers_override'),
        )

        return jsonify({'success': True, 'dataset': dataset_name or 'custom', **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/activation/functions', methods=['GET'])
def activation_functions_list():
    """Return list of activation cards/info."""
    try:
        return jsonify({'success': True, 'activations': list_activation_cards()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/activation/analyze', methods=['POST'])
def activation_analyze():
    """Analyze manual activation behavior under user-defined parameters."""
    try:
        data = request.json or {}

        input_values = data.get('input_values')
        if input_values is not None:
            input_values = [float(v) for v in input_values]

        result = analyze_activation(
            activation_name=str(data.get('activation', '')).strip().lower(),
            weight=float(data.get('weight', 1.0)),
            bias=float(data.get('bias', 0.0)),
            alpha=float(data.get('alpha', 0.01)),
            x_min=float(data.get('x_min', -6.0)),
            x_max=float(data.get('x_max', 6.0)),
            points=int(data.get('points', 241)),
            input_values=input_values,
        )

        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/digit_project/config', methods=['GET'])
def digit_project_config():
    """Return default configuration for MNIST digit project."""
    try:
        return jsonify({'success': True, **project_config()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/digit_project/train', methods=['POST'])
def digit_project_train():
    """Train binary and multiclass single-layer MNIST project models."""
    try:
        data = request.json or {}

        result = train_project(
            binary_activation=str(data.get('binary_activation', 'sigmoid')).strip().lower(),
            target_digit=int(data.get('target_digit', 0)),
            train_limit=int(data.get('train_limit', 9000)),
            test_limit=int(data.get('test_limit', 2000)),
            binary_epochs=int(data.get('binary_epochs', 35)),
            multiclass_epochs=int(data.get('multiclass_epochs', 25)),
            learning_rate=float(data.get('learning_rate', 0.08)),
            alpha_leaky=float(data.get('alpha_leaky', 0.01)),
            random_state=int(data.get('random_state', 42)),
        )

        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/digit_project/predict', methods=['POST'])
def digit_project_predict():
    """Predict an uploaded image using latest trained MNIST project models."""
    try:
        data = request.json or {}
        result = predict_uploaded_image(image_base64=str(data.get('image_base64', '')).strip())
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cnn/convolution', methods=['POST'])
def cnn_convolution_demo():
    """Perform convolution demo."""
    try:
        data = request.json or {}
        result = perform_convolution_demo(
            image_type=str(data.get('image_type', 'edges')).lower(),
            filter_type=str(data.get('filter_type', 'Vertical Edges')).lower(),
            stride=int(data.get('stride', 1)),
            padding=int(data.get('padding', 1)),
        )
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cnn/pooling', methods=['POST'])
def cnn_pooling_demo():
    """Perform pooling demo."""
    try:
        data = request.json or {}
        result = perform_pooling_demo(
            image_type=str(data.get('image_type', 'edges')).lower(),
            pool_type=str(data.get('pool_type', 'max')).lower(),
            pool_size=int(data.get('pool_size', 2)),
        )
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cnn/activation', methods=['POST'])
def cnn_activation_demo():
    """Demonstrate activation functions."""
    try:
        data = request.json or {}
        result = perform_activation_demo(
            activation_type=str(data.get('activation_type', 'relu')).lower(),
        )
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ==================== NEW IMAGE PROCESSING ROUTES ====================

@app.route('/api/cnn/image-basics', methods=['POST'])
def cnn_image_basics():
    """Process and visualize image tensor."""
    try:
        data = request.json or {}
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process image
        from cnn import process_image_upload, visualize_image_tensor
        result = process_image_upload(image_base64)
        if 'error' in result:
            return jsonify(result), 400
        
        # Visualize
        viz = visualize_image_tensor(result['image'])
        return jsonify({'success': True, **viz})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cnn/rgb-channels', methods=['POST'])
def cnn_rgb_channels():
    """Visualize RGB channels."""
    try:
        data = request.json or {}
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        from cnn import process_image_upload, visualize_rgb_channels
        result = process_image_upload(image_base64)
        if 'error' in result:
            return jsonify(result), 400
        
        viz = visualize_rgb_channels(result['image'])
        return jsonify({'success': True, **viz})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cnn/conv-magic', methods=['POST'])
def cnn_conv_magic():
    """Apply convolution with selected filter."""
    try:
        data = request.json or {}
        image_base64 = data.get('image')
        filter_type = data.get('filter_type', 'Vertical Edges')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        from cnn import apply_convolution_on_upload
        result = apply_convolution_on_upload(image_base64, filter_type)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cnn/multiple-filters', methods=['POST'])
def cnn_multiple_filters():
    """Apply multiple filters to create feature maps."""
    try:
        data = request.json or {}
        image_base64 = data.get('image')
        num_filters = int(data.get('num_filters', 4))
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        from cnn import apply_multiple_filters
        result = apply_multiple_filters(image_base64, num_filters)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cnn/relu-comparison', methods=['POST'])
def cnn_relu_comparison():
    """Compare before and after ReLU."""
    try:
        data = request.json or {}
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        from cnn import apply_relu_comparison
        result = apply_relu_comparison(image_base64)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cnn/pooling-comparison', methods=['POST'])
def cnn_pooling_comparison():
    """Apply and compare pooling types."""
    try:
        data = request.json or {}
        image_base64 = data.get('image')
        pool_type = data.get('pool_type', 'max')
        pool_size = int(data.get('pool_size', 2))
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        from cnn import apply_pooling_on_upload
        result = apply_pooling_on_upload(image_base64, pool_type, pool_size)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cnn/image-conv-lab', methods=['POST'])
def cnn_image_conv_lab():
    """Interactive image understanding lab with convolution step tracing."""
    try:
        data = request.json or {}
        image_base64 = str(data.get('image', '')).strip()
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400

        result = perform_image_convolution_lab(
            image_base64=image_base64,
            kernel_mode=str(data.get('kernel_mode', 'edge detection')),
            custom_kernel=data.get('custom_kernel'),
            stride=int(data.get('stride', 1)),
            padding=int(data.get('padding', 1)),
            num_filters=int(data.get('num_filters', 1)),
            resize_to=int(data.get('resize_to', 224)),
            preprocess_type=str(data.get('preprocess_type', 'normalize')),
        )
        if 'error' in result:
            return jsonify(result), 400
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/opencv/detect-image', methods=['POST'])
def opencv_detect_image():
    """Run YOLO object detection on an uploaded image payload."""
    try:
        data = request.json or {}
        image_base64 = str(data.get('image', '')).strip()
        confidence = float(data.get('confidence', 0.4))
        max_dim = int(data.get('max_dim', 960))
        person_only = bool(data.get('person_only', False))
        result = detect_objects_from_base64(
            image_base64=image_base64,
            confidence=confidence,
            max_dim=max_dim,
            person_only=person_only,
            return_annotated=True,
        )
        if 'error' in result:
            return jsonify(result), 400
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/opencv/detect-webcam-frame', methods=['POST'])
def opencv_detect_webcam_frame():
    """Run YOLO object detection on a webcam frame payload."""
    try:
        data = request.json or {}
        image_base64 = str(data.get('frame', '')).strip()
        confidence = float(data.get('confidence', 0.4))
        max_dim = int(data.get('max_dim', 640))
        person_only = bool(data.get('person_only', False))
        result = detect_objects_from_base64(
            image_base64=image_base64,
            confidence=confidence,
            max_dim=max_dim,
            person_only=person_only,
            return_annotated=False,
        )
        if 'error' in result:
            return jsonify(result), 400
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/rnn/train', methods=['POST'])
def rnn_train():
    """Train RNN model with specified hyperparameters."""
    try:
        data = request.json or {}
        hidden_size = int(data.get('hidden_size', 8))
        learning_rate = float(data.get('learning_rate', 0.01))
        num_epochs = int(data.get('num_epochs', 50))
        
        result = train_rnn_model(
            num_epochs=num_epochs,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            seq_length=15
        )
        
        # Calculate improvement percentage
        if result['initial_loss'] > 0:
            improvement = (1 - result['final_loss'] / result['initial_loss']) * 100
        else:
            improvement = 0
        
        result['improvement'] = improvement
        
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/rnn/architecture', methods=['GET'])
def rnn_architecture():
    """Get RNN architecture information."""
    try:
        info = get_rnn_architecture_info()
        return jsonify({'success': True, **info})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Global variables to store trained sentiment model
sentiment_models = {}
NOTEBOOK_SENTIMENT_PKL = 'rnn_project/notebook_sentiment_model.pkl'


def _ensure_default_notebook_sentiment_model():
    """Lazily load default notebook-style model from PKL; train+save only once if missing."""
    default_model_id = 'notebook_default_model'
    if default_model_id in sentiment_models:
        return {'success': True, 'model_id': default_model_id}

    if os.path.exists(NOTEBOOK_SENTIMENT_PKL):
        loaded = load_notebook_style_model_from_pkl(NOTEBOOK_SENTIMENT_PKL)
        if loaded.get('success'):
            sentiment_models[default_model_id] = {
                'mode': 'notebook',
                'model': loaded['model_object'],
                'tokenizer': loaded['tokenizer'],
                'max_len': loaded['max_len'],
                'label_map': loaded['label_map'],
                'meta': loaded.get('meta', {}),
            }
            return {'success': True, 'model_id': default_model_id}

    boot = train_notebook_style_model(epochs=10)
    if not boot.get('success'):
        return {
            'success': False,
            'error': boot.get('error', 'Failed to initialize default notebook model.'),
        }

    save_result = save_notebook_style_model_to_pkl(boot, NOTEBOOK_SENTIMENT_PKL)
    if (not save_result.get('success')):
        return {
            'success': False,
            'error': save_result.get('error', 'Failed to persist notebook model to PKL.'),
        }

    sentiment_models[default_model_id] = {
        'mode': 'notebook',
        'model': boot['model_object'],
        'tokenizer': boot['tokenizer'],
        'max_len': boot['max_len'],
        'label_map': boot['label_map'],
        'meta': {
            'train_samples': boot.get('train_samples'),
            'test_samples': boot.get('test_samples'),
            'epochs': boot.get('epochs'),
            'test_accuracy': boot.get('test_accuracy'),
            'test_loss': boot.get('test_loss'),
            'pkl_path': NOTEBOOK_SENTIMENT_PKL,
        },
    }
    return {'success': True, 'model_id': default_model_id}


@app.route('/api/sentiment/train', methods=['POST'])
def sentiment_train():
    """Train sentiment analysis model with custom parameters."""
    try:
        data = request.json or {}
        
        # Get parameters
        dataset_path = data.get('dataset_path', 'rnn_project/testdata.manual.2009.06.14.csv')
        epochs = int(data.get('epochs', 20))
        batch_size = int(data.get('batch_size', 16))
        vocab_size = int(data.get('vocab_size', 5000))
        embedding_dim = int(data.get('embedding_dim', 64))
        lstm_units = int(data.get('lstm_units', 128))
        dense_units = int(data.get('dense_units', 64))
        max_len = int(data.get('max_len', 100))
        test_split = float(data.get('test_split', 0.2))
        
        # Train model
        result = train_sentiment_model(
            csv_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            dense_units=dense_units,
            max_len=max_len,
            test_split=test_split
        )
        
        if not result['success']:
            return jsonify(result), 400
        
        # Store model for later use
        model_id = f"sentiment_model_{len(sentiment_models)}"
        sentiment_models[model_id] = {
            'mode': 'lstm',
            'model': result.pop('model_object'),
            'tokenizer': result.pop('tokenizer'),
            'label_encoder': result.pop('label_encoder'),
            'model_info': result.get('model_info', {}),
            'max_len': max_len
        }
        
        result['model_id'] = model_id
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/sentiment/predict', methods=['POST'])
def sentiment_predict():
    """Predict sentiment using trained model."""
    try:
        data = request.json or {}
        text = data.get('text', '').strip()
        model_id = data.get('model_id')
        
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        
        if not model_id:
            ensure = _ensure_default_notebook_sentiment_model()
            if not ensure.get('success'):
                return jsonify(ensure), 400
            model_id = ensure['model_id']

        if model_id not in sentiment_models:
            return jsonify({'success': False, 'error': 'Model not found.'}), 400
        
        # Get model
        model_data = sentiment_models[model_id]
        
        if model_data.get('mode') == 'notebook':
            result = predict_sentiment_notebook_style(
                text=text,
                model=model_data['model'],
                tokenizer=model_data['tokenizer'],
                label_map=model_data['label_map'],
                max_len=model_data['max_len'],
            )
        else:
            result = predict_sentiment(
                text=text,
                model=model_data['model'],
                tokenizer=model_data['tokenizer'],
                label_encoder=model_data['label_encoder'],
                max_len=model_data['max_len'],
            )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/sentiment/model-info', methods=['GET'])
def sentiment_model_info():
    """Return notebook-trained model details for dashboard display."""
    try:
        ensure = _ensure_default_notebook_sentiment_model()
        if not ensure.get('success'):
            return jsonify(ensure), 400

        model_id = ensure['model_id']
        model_data = sentiment_models.get(model_id, {})
        meta = model_data.get('meta', {})

        return jsonify({
            'success': True,
            'model_id': model_id,
            'model_name': 'Notebook SimpleRNN Sentiment Classifier',
            'source_notebook': 'rnn_project/sentiment-analysis-using-simple-rnn.ipynb',
            'artifact': {
                'format': 'pkl',
                'path': NOTEBOOK_SENTIMENT_PKL,
                'loaded_mode': model_data.get('mode', 'notebook'),
            },
            'dataset': {
                'primary': 'train.csv / test.csv (if available)',
                'fallback': 'rnn_project/testdata.manual.2009.06.14.csv',
                'classes': ['positive', 'negative', 'neutral'],
            },
            'tokenizer': {
                'num_words': 20000,
                'padding': 'post',
                'max_len': 35,
            },
            'compile': {
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy'],
            },
            'architecture': [
                {'layer': 'Embedding', 'config': 'input_dim=20000, output_dim=2, input_length=35'},
                {'layer': 'SimpleRNN', 'config': 'units=32, return_sequences=False'},
                {'layer': 'Dense', 'config': 'units=3, activation=softmax'},
            ],
            'build_steps': [
                'Load sentiment dataset and keep text + sentiment columns',
                'Handle missing text values',
                'Map labels to classes: positive=0, negative=1, neutral=2',
                'Convert labels to one-hot vectors using to_categorical',
                'Fit tokenizer with num_words=20000 on text corpus',
                'Transform text to token sequences',
                'Pad sequences to fixed length max_len=35',
                'Build model: Embedding -> SimpleRNN -> Dense(softmax)',
                'Compile with adam and categorical_crossentropy',
                'Train with validation set and use model for inference',
            ],
            'runtime_metrics': {
                'train_samples': meta.get('train_samples'),
                'test_samples': meta.get('test_samples'),
                'epochs': meta.get('epochs'),
                'test_accuracy': meta.get('test_accuracy'),
                'test_loss': meta.get('test_loss'),
            },
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
