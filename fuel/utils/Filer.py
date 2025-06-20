import os
import shutil
from pathlib import Path


class File:
    lib = "pytorch"
    # basic dir paths
    root_dir, input_dir, output_dir, test_dir = "", "", "", ""
    cur_filename = ""

    rendered_code, eliminated_code = "", ""

    # running time files
    tmp_py = ""
    validate_file = ""
    err_file, total_errs_file = "", ""
    fail_file, success_file = "", ""
    crash_file, skip_file = "", ""
    bug_file, bug_report = "", ""
    als_file, gen_file = "", ""

    res_base_file, res_target_file = "", ""

    cov_rc_file = ""
    cov_base_file, cov_target_file = "", ""

    feedback_file, ops_file, fix_file = "", "", ""

    @classmethod
    def init(cls, root_dir, input_dir, output_dir, lib):
        cls.lib = lib
        cls.set_root_dir(root_dir)
        cls.set_input_dir(os.path.join(input_dir, lib))
        cls.set_output_dir(output_dir)
        cls.set_test_dir(lib)
        cls.set_output_file()
        cls.set_coverage_file()
        cls.get_library_path()

    @classmethod
    def set_input_dir(cls, folder_name: str):
        cls.input_dir = os.path.join(cls.root_dir, folder_name)
        if Path(cls.input_dir).exists():
            shutil.rmtree(Path(cls.input_dir))
        Path(cls.input_dir).mkdir(parents=True)

    @classmethod
    def set_output_dir(cls, folder_name):
        cls.output_dir = os.path.join(cls.root_dir, folder_name)
        if not os.path.exists(cls.output_dir):
            os.mkdir(cls.output_dir)

    @classmethod
    def set_test_dir(cls, lib):
        cls.test_dir = os.path.join(cls.output_dir, lib)
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)
        os.mkdir(cls.test_dir)

    @classmethod
    def set_output_file(cls):
        cls.total_errs_file = os.path.join(cls.test_dir, "total_errs.log")
        cls.err_file = os.path.join(cls.test_dir, "err.log")
        cls.success_file = os.path.join(cls.test_dir, "success.log")
        cls.fail_file = os.path.join(cls.test_dir, "fail.log")
        cls.validate_file = os.path.join(cls.test_dir, "validate.log")
        cls.tmp_py = os.path.join(cls.test_dir, "tmp.py")
        cls.res_base_file = os.path.join(cls.test_dir, "res_base.bin")
        cls.res_target_file = os.path.join(cls.test_dir, "res_target.bin")
        cls.bug_file = os.path.join(cls.test_dir, "bug.log")
        cls.bug_report = os.path.join(cls.test_dir, "bug_report.log")
        cls.als_file = os.path.join(cls.test_dir, "als.log")
        cls.gen_file = os.path.join(cls.test_dir, "gen.log")
        cls.skip_file = os.path.join(cls.test_dir, "skip.log")
        cls.ops_file = os.path.join(cls.test_dir, "ops.log")
        cls.feedback_file = os.path.join(cls.test_dir, "feedback.log")
        cls.fix_file = os.path.join(cls.test_dir, "fix.log")

    @classmethod
    def set_coverage_file(cls):
        cls.cov_base_file = os.path.join(cls.test_dir, "base.coverage")
        cls.cov_target_file = os.path.join(cls.test_dir, "target.coverage")
        cls.cov_rc_file = os.path.join(cls.test_dir, ".coveragerc")
        with open(cls.cov_rc_file, "w") as f:
            f.write("[run]\nsource = ${TESTED_LIB_PATH}")

    @classmethod
    def remove(cls, filename):
        if not os.path.exists(filename):
            return
        if os.path.isdir(filename):
            try:
                shutil.rmtree(filename)
            except Exception as e:
                print(e)
                print(
                    "The folder deletion failed, possibly due to it being in use or insufficient permissions."
                )
        else:
            os.remove(filename)

    @classmethod
    def write_file(cls, filename, content, mode="a+"):
        with open(filename, mode, encoding="utf-8") as f:
            f.write(content + "\n")

    @classmethod
    def read_file(cls, filename):
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    @classmethod
    def open_file(cls, filename, mode="a+"):
        return open(filename, mode)

    @classmethod
    def close_file(cls, file):
        with open(file) as f:
            f.close()

    @classmethod
    def get_library_path(cls):
        if cls.lib == "pytorch":
            import torch

            lib_path = os.path.dirname(torch.__file__)
        elif cls.lib == "tensorflow":
            import tensorflow as tf

            lib_path = os.path.dirname(tf.__file__)
        else:
            raise ValueError("Invalid library name")
        os.environ["TESTED_LIB_PATH"] = lib_path
        return lib_path

    @classmethod
    def set_root_dir(cls, root_dir):
        cls.root_dir = root_dir
