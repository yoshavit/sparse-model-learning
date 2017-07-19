from tensorflow.python.training.saver import get_checkpoint_state, _prefix_to_checkpoint_path, file_io, logging, saver_pb2
import re
def latest_checkpoint(checkpoint_dir, latest_filename=None):
  """Finds the filename of latest saved checkpoint file.
  Args:
    checkpoint_dir: Directory where the variables were saved.
    latest_filename: Optional name for the protocol buffer file that
      contains the list of most recent checkpoint filenames.
      See the corresponding argument to `Saver.save()`.
  Returns:
    The full path to the latest checkpoint or `None` if no checkpoint was found.
  """
  # Pick the latest checkpoint based on checkpoint state.
  ckpt = get_checkpoint_state(checkpoint_dir, latest_filename)
  # strips directory-identifying component from path
  suffix = re.search("model\.ckpt.+", ckpt.model_checkpoint_path).group(0)
  print(ckpt.model_checkpoint_path)
  print(suffix)
  print(checkpoint_dir)
  path = checkpoint_dir + suffix
  if ckpt and path:
    # Look for either a V2 path or a V1 path, with priority for V2.
    v2_path = _prefix_to_checkpoint_path(path,
                                         saver_pb2.SaverDef.V2)
    v1_path = _prefix_to_checkpoint_path(path,
                                         saver_pb2.SaverDef.V1)
    if file_io.get_matching_files(v2_path) or file_io.get_matching_files(
        v1_path):
      return path
    else:
      logging.error("Couldn't match files for checkpoint %s",
                    ckpt.model_checkpoint_path)
  return None
