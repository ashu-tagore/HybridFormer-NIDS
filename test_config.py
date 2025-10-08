import yaml

config = yaml.safe_load(open('configs/hybridformer.yaml'))
print('Config loaded successfully')
print(f"Model: {config['model']['name']}")
print(f"Batch size: {config['data']['batch_size']}")
print(f"Learning rate: {config['training']['optimizer']['lr']}")