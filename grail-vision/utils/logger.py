import json
import csv
import time
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    """
    Logger - Saves a JSON file for each model containing data for all compression ratios
    """
    
    def __init__(self, experiment_name=None, log_dir="./experiment_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(exist_ok=True)
        
        self.experiment_data = {
            'metadata': {
                'experiment_name': experiment_name,
                'start_time': datetime.now().isoformat(),
                'method': None,
                'dataset': None
            },
            'ratios': {},  # Store data for different compression ratios
            'overall_summary': []
        }
    
    def log_metadata(self, method, dataset="cifar10"):
        """Log experiment metadata"""
        self.experiment_data['metadata'].update({
            'method': method,
            'dataset': dataset
        })
    
    def log_ratio_data(self, ratio, original_acc, compre_acc, compre_re_acc, compre_re_compen_acc, compre_re_compen_re_acc, compre_compen_acc, compre_compen_re_acc,
                      original_params, compressed_params, layer_details=None):
        """Log data for specific compression ratio"""
        ratio_key = f"ratio_{ratio}"
        
        self.experiment_data['ratios'][ratio_key] = {
            'compression_ratio': ratio,
            'performance': {
                'original_accuracy': original_acc,
                'compressed_accuracy': compre_acc,
                'compressed_repaired_accuracy': compre_re_acc,
                'compressed_repaired_compensated_accuracy': compre_re_compen_acc,
                'compressed_repaired_compensated_repaired_accuracy': compre_re_compen_re_acc,
                'compressed_compensated_accuracy': compre_compen_acc,
                'compressed_compensated_repaired_accuracy': compre_compen_re_acc,
                'original_params': original_params,
                'compressed_params': compressed_params,
                'compression_rate': (original_params - compressed_params) / original_params
            },
            'layer_details': layer_details or []
        }
        
        # Also add to overall summary
        summary_entry = {
            'compression_ratio': ratio,
            'original_accuracy': original_acc,
            'compressed_accuracy': compre_acc,
            'compressed_repaired_accuracy': compre_re_acc,
            'compressed_repaired_compensated_accuracy': compre_re_compen_acc,
            'compressed_repaired_compensated_repaired_accuracy': compre_re_compen_re_acc,
            'compressed_compensated_accuracy': compre_compen_acc,
            'compressed_compensated_repaired_accuracy': compre_compen_re_acc,
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_rate': (original_params - compressed_params) / original_params
        }
        self.experiment_data['overall_summary'].append(summary_entry)
    
    def log_layer_details(self, ratio, layer_name, original_channels, pruned_channels, 
                         compensation_applied=False, compensation_type=None):
        """Log detailed layer information for specific compression ratio"""
        ratio_key = f"ratio_{ratio}"
        
        if ratio_key not in self.experiment_data['ratios']:
            self.experiment_data['ratios'][ratio_key] = {
                'compression_ratio': ratio,
                'performance': {},
                'layer_details': []
            }
        
        layer_info = {
            'layer_name': layer_name,
            'original_channels': original_channels,
            'pruned_channels': pruned_channels,
            'compression_rate': (original_channels - pruned_channels) / original_channels,
            'compensation_applied': compensation_applied,
            'compensation_type': compensation_type
        }
        self.experiment_data['ratios'][ratio_key]['layer_details'].append(layer_info)
    
    def log_timing(self, stage, duration):
        """Log timing information"""
        if 'timing' not in self.experiment_data:
            self.experiment_data['timing'] = {}
        self.experiment_data['timing'][stage] = duration
    
    def save_json(self):
        """Save JSON data file"""
        self.experiment_data['metadata']['end_time'] = datetime.now().isoformat()
        
        json_path = self.exp_dir / "experiment_data.json"
        with open(json_path, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
        print(f"[INFO] Experiment data saved to: {json_path}")
        return json_path
    
    def save_csv_summary(self):
        """Save CSV summary file"""
        csv_path = self.exp_dir / "experiment_summary.csv"
        
        if not self.experiment_data['overall_summary']:
            print("[WARNING] No summary data to save")
            return csv_path
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = self.experiment_data['overall_summary'][0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.experiment_data['overall_summary'])
        
        print(f"[INFO] Summary saved to: {csv_path}")
        return csv_path
    
    def save_layer_details_csv(self):
        """Save layer details to CSV"""
        csv_path = self.exp_dir / "layer_details.csv"
        
        all_layer_details = []
        for ratio_key, ratio_data in self.experiment_data['ratios'].items():
            for layer_detail in ratio_data.get('layer_details', []):
                layer_detail_with_ratio = layer_detail.copy()
                layer_detail_with_ratio['compression_ratio'] = ratio_data['compression_ratio']
                all_layer_details.append(layer_detail_with_ratio)
        
        if all_layer_details:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_layer_details[0].keys())
                writer.writeheader()
                writer.writerows(all_layer_details)
            print(f"[INFO] Layer details saved to: {csv_path}")
        
        return csv_path
    
    def get_ratio_data(self, ratio):
        """Get data for specific compression ratio"""
        ratio_key = f"ratio_{ratio}"
        return self.experiment_data['ratios'].get(ratio_key)
    
    def get_all_ratios(self):
        """Get list of all compression ratios"""
        return [data['compression_ratio'] for data in self.experiment_data['ratios'].values()]
    
    def get_performance_summary(self):
        """Get performance summary"""
        return self.experiment_data['overall_summary']


def test_logger():
    """Test the new logger functionality"""
    logger = ExperimentLogger("test_model_wanda")
    logger.log_metadata("wanda", "cifar10")
    
        # Simulate data for different compression ratios
    test_data = [
        (0.0, 92.5, 92.5, 92.5, 11173962, 11173962),
        (0.1, 92.5, 85.3, 89.2, 11173962, 10056565),
        (0.2, 92.5, 78.1, 85.7, 11173962, 8939170),
    ]
    
    for ratio, orig_acc, pruned_acc, comp_acc, orig_params, comp_params in test_data:
        logger.log_ratio_data(ratio, orig_acc, pruned_acc, comp_acc, orig_params, comp_params)
        
        # Add some layer details
        logger.log_layer_details(ratio, "layer1.conv1", 64, 58)
        logger.log_layer_details(ratio, "layer1.conv2", 64, 58)
    
    # Save all data
    logger.save_json()
    logger.save_csv_summary()
    logger.save_layer_details_csv()
    
    print("Logger test completed successfully!")


if __name__ == "__main__":
    test_logger()
