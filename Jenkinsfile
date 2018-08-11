
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
virtualenv -p python3.5 .venv
. .venv/bin/activate

# Install dev dependencies
pip3 install -e '.[dev]'

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
