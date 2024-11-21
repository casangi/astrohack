#!/usr/bin/bash

echo 'Astrohack environment creation and installation script'
echo 

location=${HOME}'/.local/python/venvs'
echo -e 'Default installation in '${location}
echo 'Press <enter> to continue or type desired location for installation.'
read user_location
if [ -z "${user_location}" ]; then
    echo 'Using '${location}' for installation'
else
    echo -e 'Using '${user_location}' for installation'
    location=${user_location}
fi
echo 


pyexec="$(which python3.11)"
echo 'Default python executable is '${pyexec}
echo 'Press <enter> to continue or type desired python executable'
read user_pyexec
if [ -z "${user_pyexec}" ]; then
    echo 'Using '${pyexec}' as the python executable'
else
    echo 'Using '${user_pyexec}' as the python executable'
    pyexec=${user_pyexec}
fi
echo 

envname="astrohack-py3.11"
echo 'Default environment name is '${envname}
echo 'Press <enter> to continue or type desired environment name'
read user_envname
if [ -z "${user_envname}" ]; then
    echo 'Using '${envname}' as environment name'
else
    echo 'Using '${user_envname}' as environment name'
    envname=${user_envname}
fi
echo 

# actual installation steps
env_address=${location}/${envname}
echo 'Creating environment...'
eval "${pyexec} -m venv --clear ${env_address}"
echo

eval "source ${env_address}/bin/activate"
echo 'Updating pip...'
eval "pip install --upgrade pip 1> /dev/null"
echo

echo 'Installing astrohack...'
eval "pip install astrohack 1> /dev/null"
echo

echo 'Add convenience functions <activate_astrohack> and <get_locit_scripts> to .profile?'
echo '<y/enter/n>'
read yesno
if [ -z "${yesno}" ] || [ "${yesno}" = "y" ]; then
    echo 'Adding convenience functions to .profile'
    reset_shell='yes'

    cp ${HOME}/.profile ${HOME}/.profile.copy
    echo >> ${HOME}/.profile
    echo '# Astrohack convenience functions:' >> ${HOME}/.profile
    echo >> ${HOME}/.profile
    echo 'activate_astrohack () {' >> ${HOME}/.profile
    echo '    source '${env_address}'/bin/activate' >> ${HOME}/.profile
    echo '}' >> ${HOME}/.profile
    echo  >> ${HOME}/.profile
    echo 'get_locit_scripts () {' >> ${HOME}/.profile
    echo '    echo Downloading CASA pre locit script...' >> ${HOME}/.profile
    echo '    wget https://github.com/casangi/astrohack/raw/main/etc/casa/pre-locit-script.py' >> ${HOME}/.profile
    echo '    echo' >> ${HOME}/.profile
    echo '    echo Downloading astrohack locit script...' >> ${HOME}/.profile
    echo '    echo wget https://github.com/casangi/astrohack/raw/main/etc/casa/pre-locit-script.py' >> ${HOME}/.profile
    echo '}' >> ${HOME}/.profile
    
else
    echo 'NOPE!'
    reset_shell='no'
fi
echo

if [ "${reset_shell}" = 'yes' ]; then
    echo 'Previous .profile saved as .profile.copy'
    echo 'Please restart shell to access convenience functions'
fi
