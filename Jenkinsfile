
node {
  stage('clone') {
    checkout scm
  }
  stage('build') {
    def commitId = sh(returnStdout: true, script: 'git rev-parse HEAD')

    bitbucketStatusNotify(
      buildState: 'INPROGRESS',
      commitId: commitId
    )


    try {
    sh """
#Create and enable venv
source activate trading

# Install dev dependencies
python setup.py install
pip install -r requirements.txt

# Run the tests
pytest
"""
      bitbucketStatusNotify(
        buildState: 'SUCCESSFUL',
        commitId: commitId
      )
    } catch(Exception e) {
      bitbucketStatusNotify(
        buildState: 'FAILED',
        commitId: commitId
      )
      throw e
    }
  }
}
