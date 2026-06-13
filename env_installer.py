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

def create_clean_envs():
    for yml in clean_envs:
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

def create_rascore_env():
    print("🔹 Creating rascore-env manually ...")
    try:
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

        print("✅ Successfully created rascore-env")
    except subprocess.CalledProcessError:
        print("❌ Failed to create rascore-env")
        sys.exit(1)



def main():
    print("Starting environment installation...")
    create_clean_envs()
    create_rascore_env()
    print("\nAll environments installed successfully!")

if __name__ == "__main__":
    main()
