# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                                                         *
# * Author: Riccardo Finotello <riccardo.finotello@cea.fr>  *
# * Date:   2024-11-07                                      *
# *                                                         *
# * Environment variables for scripts and tools             *
# *                                                         *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

# Resolve the directory where this script is located
export FRG_PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Detect if we are in a developer environment (with src/frg) or user workspace
if [ -d "${FRG_PROJECT_ROOT}/src/frg" ]; then
    # Developer mode
    export FRG_CONFIGPATH=${FRG_PROJECT_ROOT}/src/frg/configs
    export FRG_SCRIPTPATH=${FRG_PROJECT_ROOT}/src/frg/scripts
else
    # User workspace mode (after frg-init)
    export FRG_CONFIGPATH=${FRG_PROJECT_ROOT}/configs
    export FRG_SCRIPTPATH=${FRG_PROJECT_ROOT}/scripts
fi

export SCRATCHDIR=${SCRATCHDIR:-${HOME}/Datasets}
