<<<<<<< HEAD

=======
import os
import shutil
from sklearn.model_selection import train_test_split
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def resolve_dataset_dir(custom_dir: str | None = None) -> str:
  """Mencari direktori Dataset yang valid baik dari root proyek maupun subfolder model/mobilenetv2."""
  candidates = []
  if custom_dir:
    candidates.append(custom_dir)

  this_dir = os.path.dirname(os.path.abspath(__file__))
  candidates.extend([
      './Dataset',
      '../../Dataset',
      os.path.abspath(os.path.join(this_dir, '..', '..', 'Dataset')),
      r'F:\COLLEGE\semester 3\ipsd\tugasBesar\Dataset',
  ])

  for path in candidates:
    abs_path = os.path.abspath(path)
    if os.path.isdir(abs_path):
      return abs_path

  raise FileNotFoundError(
      'Folder Dataset tidak ditemukan. Periksa lokasi folder Dataset pada proyek.'
  )


def get_mobilenet_v2(num_classes: int, pretrained: bool = True):
  """Membuat model MobileNetV2 ringan dari timm."""
  model = timm.create_model(
      'mobilenetv2_100', pretrained=pretrained, num_classes=num_classes
  )
  return model


def _find_class_root(source_dir: str) -> str:
  """Mencari folder yang berisi subfolder kelas gambar."""
  for root, _, files in os.walk(source_dir):
    imgs = [
        f
        for f in files
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]
    if imgs:
      return os.path.dirname(root)
  raise FileNotFoundError(
      f'Tidak ditemukan folder gambar di dalam: {source_dir}'
  )


def prepare_local_split(
    raw_dir: str,
    prepared_dir: str = './split_data',
    test_size: float = 0.2,
    seed: int = 42,
) -> str:
  """Jika raw_dir belum punya train/ dan test/, pecah otomatis ke prepared_dir."""
  if os.path.exists(os.path.join(prepared_dir, 'train')) and os.path.exists(
      os.path.join(prepared_dir, 'test')
  ):
    return prepared_dir

  if os.path.exists(os.path.join(raw_dir, 'train')) and os.path.exists(
      os.path.join(raw_dir, 'test')
  ):
    return raw_dir

  class_root = _find_class_root(raw_dir)
  class_folders = [
      d
      for d in os.listdir(class_root)
      if os.path.isdir(os.path.join(class_root, d))
  ]

  for cls_name in class_folders:
    cls_path = os.path.join(class_root, cls_name)
    imgs = [
        f
        for f in os.listdir(cls_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]
    if len(imgs) < 2:
      continue

    train_imgs, test_imgs = train_test_split(
        imgs, test_size=test_size, random_state=seed, shuffle=True
    )

    os.makedirs(os.path.join(prepared_dir, 'train', cls_name), exist_ok=True)
    os.makedirs(os.path.join(prepared_dir, 'test', cls_name), exist_ok=True)

    for img in train_imgs:
      shutil.copy2(
          os.path.join(cls_path, img),
          os.path.join(prepared_dir, 'train', cls_name, img),
      )
    for img in test_imgs:
      shutil.copy2(
          os.path.join(cls_path, img),
          os.path.join(prepared_dir, 'test', cls_name, img),
      )

  return prepared_dir


def get_data_loaders(
    data_dir: str | None = None,
    batch_size: int = 16,
    img_size: int = 96,
    prepared_dir: str = './split_data',
    num_workers: int = 0,
):
  """Menyiapkan DataLoader train dan test dengan resolusi ringan untuk CPU."""
  resolved_dir = resolve_dataset_dir(data_dir)
  active_dir = prepare_local_split(resolved_dir, prepared_dir=prepared_dir)

  train_transforms = transforms.Compose([
      transforms.Resize((img_size, img_size)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  test_transforms = transforms.Compose([
      transforms.Resize((img_size, img_size)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  train_dataset = datasets.ImageFolder(
      os.path.join(active_dir, 'train'), transform=train_transforms
  )
  test_dataset = datasets.ImageFolder(
      os.path.join(active_dir, 'test'), transform=test_transforms
  )

  train_loader = DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
  )
  test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
  )

  return {'train': train_loader, 'test': test_loader}, train_dataset.classes
>>>>>>> f0285b9 (model mobilenet)
