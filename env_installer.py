import glob
import json
import os
import subprocess
import sys

# List of clean envs with working .yml files
clean_envs = [
    "environments/sascore-env.yml",
    "environments/scscore-env.yml",
    "environments/syba-env.yml",
    "environments/chemprop-env.yml",
    "environments/fsscore-env.yml"
]


def existing_envs():
    """Return the set of conda environment names already present."""
    out = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=True, capture_output=True, text=True,
    ).stdout
    names = set()
    for prefix in json.loads(out).get("envs", []):
        names.add(prefix.rstrip("/").rsplit("/", 1)[-1])
    return names


def env_name_from_yml(yml):
    with open(yml) as f:
        for line in f:
            if line.strip().startswith("name:"):
                return line.split(":", 1)[1].strip()
    return None


def create_clean_envs(present):
    for yml in clean_envs:
        name = env_name_from_yml(yml)
        if name in present:
            print(f"⏭️  {name} already exists — skipping {yml}")
            continue
        print(f"🔹 Creating env from {yml} ...")
        try:
            subprocess.run(
                ["conda", "env", "create", "-f", yml],
                check=True
            )
            print(f"✅ Created env from {yml}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to create env from {yml}")
            sys.exit(1)

def env_prefix(name):
    """Return the filesystem prefix of a conda env, or None if absent."""
    out = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=True, capture_output=True, text=True,
    ).stdout
    for prefix in json.loads(out).get("envs", []):
        if prefix.rstrip("/").rsplit("/", 1)[-1] == name:
            return prefix
    return None


def ensure_libgomp_soname(prefix):
    """xgboost (used by RAscore) loads libgomp.so.1, but the conda-forge libgomp
    package ships only libgomp.so.1.0.0 — not the SONAME symlink the loader
    requests. Create it so `import xgboost` works."""
    if not prefix:
        return
    lib = os.path.join(prefix, "lib")
    soname = os.path.join(lib, "libgomp.so.1")
    if os.path.exists(soname):
        return
    versioned = sorted(glob.glob(os.path.join(lib, "libgomp.so.1.*")))
    if versioned:
        target = os.path.basename(versioned[-1])
        os.symlink(target, soname)
        print(f"🔧 Linked libgomp.so.1 -> {target}")


def rascore_importable():
    """True if rascore-env exists AND can import RAscore *and* xgboost (i.e. fully
    built, including the libgomp runtime xgboost needs to load)."""
    return subprocess.run(
        ["conda", "run", "-n", "rascore-env", "python", "-c",
         "import RAscore, xgboost"],
        capture_output=True,
    ).returncode == 0


def create_rascore_env(present):
    exists = "rascore-env" in present
    if exists and rascore_importable():
        print("⏭️  rascore-env already exists and is complete — skipping")
        return
    try:
        if not exists:
            print("🔹 Creating rascore-env manually ...")
            # -------- Create bare env with Python 3.7 --------
            subprocess.run(
                ["conda", "create", "-n", "rascore-env", "python=3.7", "-y"],
                check=True
            )

            # -------- Install rdkit 2020.03 from rdkit channel --------
            subprocess.run(
                [
                    "conda", "run", "-n", "rascore-env",
                    "conda", "install", "-c", "rdkit", "rdkit=2020.03", "-y"
                ],
                check=True
            )
        else:
            # Env exists but RAscore won't import — a previous run was interrupted
            # (e.g. before the pip step finished). Re-run the pip install to complete it.
            print("🔧 rascore-env exists but RAscore is missing — completing pip install ...")

        # -------- Install the OpenMP runtime xgboost/RAscore needs --------
        # Without libgomp.so.1, `import xgboost` fails and RAscore can't score.
        subprocess.run(
            [
                "conda", "install", "-n", "rascore-env",
                "-c", "conda-forge", "libgomp", "-y"
            ],
            check=True
        )

        # -------- Install packages via pip inside env --------
        pip_packages = [
            "numpy==1.19.5",
            "scikit-learn==0.22.1",
            "tensorflow==2.5.0",
            "xgboost==1.0.2",
            "optuna",
            "git+https://github.com/reymond-group/RAscore.git@master"
        ]
        subprocess.run(
            ["conda", "run", "-n", "rascore-env", "pip", "install"] + pip_packages,
            check=True
        )

        # conda-forge libgomp omits the libgomp.so.1 SONAME symlink — add it.
        ensure_libgomp_soname(env_prefix("rascore-env"))

        print("✅ Successfully created rascore-env")
    except subprocess.CalledProcessError:
        print("❌ Failed to create rascore-env")
        sys.exit(1)



def main():
    print("Starting environment installation...")
    present = existing_envs()
    create_clean_envs(present)
    create_rascore_env(present)
    print("\nAll environments installed successfully!")

if __name__ == "__main__":
    main()
