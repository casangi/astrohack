import toml
import argparse
import os

parser = argparse.ArgumentParser(description="Bumps the current version, reintalls with pip, "
                                 "updates notebooks, then does git")
parser.add_argument('-b', '--bump', help='Type of bump to apply to version (defaults is minor)',
                    default='minor', required=False)
parser.add_argument('-p', '--push', help='Push release commit to origin', action='store_true',
                    default=False, required=False)

args = parser.parse_args()

toml_file_name = 'pyproject.toml'
notebooks = ['tutorial_vla.ipynb',
             'visualization_tutorial.ipynb',
             'locit_tutorial.ipynb']

def print_section_header(header):
    print('\n'+80*'*')
    print(header)
    print()

def bump_version(bump):
    with open(toml_file_name, 'r') as project_file:
        project_toml = toml.load(project_file)
        
    current_version = project_toml['project']['version']
    revision, major, minor = current_version.split('.')

    if bump == 'revision':
        new_version = f'{int(revision)+1}.0.0'
    elif bump == 'major':
        new_version = f'{revision}.{int(major)+1}.0'
    elif bump == 'minor':
        new_version = f'{revision}.{major}.{int(minor)+1}'
    else:
        raise Exception(f'Do not know what to do with a {bump} bump')

    project_toml['project']['version'] = new_version
    print(f"Bumping version to {new_version}")
    with open(toml_file_name, 'w') as project_file:
        toml.dump(project_toml, project_file)
    return new_version

def pip_reinstall():
    print_section_header('Running pip install...')
    os.system('pip install -e . 1>>/dev/null')

def run_notebooks():
    print_section_header('Running notebooks...')
    exestr = "python run-notebooks.py"
    for notebook in notebooks:
        exestr += ' '+notebook
    exestr += ' -o'
    os.chdir('./docs')
    os.system(exestr)
    os.system("bash < cleanup-notebooks.sh")
    os.chdir('../')

def run_git(new_version, push):
    print_section_header('Running git...')
    os.system('git add '+toml_file_name)
    for notebook in notebooks:
        os.system('git add ./docs/'+notebook)
    os.system(f'git commit -m "Bumped version to v{new_version}"')
    os.system(f'git tag -a v{new_version} -m "v{new_version}"')
    if push:
        print('Pushing...')
        os.system('git push')
    else:
        print('Push skipped')


new_version = bump_version(args.bump)
pip_reinstall()
run_notebooks()
run_git(new_version, args.push)

print_section_header('All Done!')
