import toml
import argparse
import os
import fileinput

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
        bumped_version = f'{int(revision)+1}.0.0'
    elif bump == 'major':
        bumped_version = f'{revision}.{int(major)+1}.0'
    elif bump == 'minor':
        bumped_version = f'{revision}.{major}.{int(minor)+1}'
    else:
        raise Exception(f'Do not know what to do with a {bump} bump')

    project_toml['project']['version'] = bumped_version
    print(f"Bumping version to {bumped_version}")
    with open(toml_file_name, 'w') as project_file:
        toml.dump(project_toml, project_file)
    return bumped_version


def pip_reinstall():
    print_section_header('Running pip install...')
    os.system('pip install -e . 1>>/dev/null')


def run_notebooks(notebook_list):
    print_section_header('Running notebooks...')
    exestr = "python run-notebooks.py"
    for notebook in notebook_list:
        exestr += ' '+notebook
    exestr += ' -o'
    os.chdir('./docs')
    os.system(exestr)
    os.system("bash < cleanup-notebooks.sh")
    os.chdir('../')


def run_git(bumped_version, push):
    print_section_header('Running git...')
    os.system('git add '+toml_file_name)
    for notebook in notebooks:
        os.system('git add ./docs/'+notebook)
    os.system(f'git commit -m "Bumped version to v{bumped_version}"')
    os.system(f'git tag -a v{bumped_version} -m "v{bumped_version}"')
    if push:
        print('Pushing...')
        os.system('git push')
        os.system(f'git push v{bumped_version}')
    else:
        print('Push skipped')


def updated_colab_link(notebook_list, version):
    print_section_header('Updating Colab links...')
    for notebook in notebook_list:
        for line in fileinput.input('docs/'+notebook, inplace=1):
            if "![Open In Colab]" in line:
                wrds = line.split('/')
                iblob = wrds.index('blob')
                wrds[iblob+1] = 'v'+version
                print('/'.join(wrds)[:-1])
            else:
                print(line[:-1])
                pass


new_version = bump_version(args.bump)
pip_reinstall()
updated_colab_link(notebooks, new_version)
run_notebooks(notebooks)
run_git(new_version, args.push)

print_section_header('All Done!')
