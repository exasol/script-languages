/**
 * Triggered from a message on a Cloud Pub/Sub topic.
 *
 * @param {!Object} event Event payload.
 * @param {!Object} context Metadata for the event.
 */
exports.githubBuildStatusNotification = (event, context) => {
  const  pubsubMessage = JSON.parse(Buffer.from(event.data, 'base64').toString())

  const commitSha = pubsubMessage.sourceProvenance.resolvedRepoSource.commitSha;
  const status = pubsubMessage.status;
  const githubUser = pubsubMessage.substitutions._GITHUB_USER_NAME;
  const githubRepo = pubsubMessage.substitutions._GITHUB_REPOSITORY;
  const jobType = pubsubMessage.substitutions._JOB_TYPE;
  const flavor = pubsubMessage.substitutions._FLAVOR;
  const flavors = pubsubMessage.substitutions._FLAVORS;
  const buildId =  pubsubMessage.id;

  console.log('commitSha: '+commitSha);
  console.log('status: '+status);
  console.log('jobType: '+jobType);
  console.log('flavor: '+flavor);
  console.log('flavors: '+flavors);
  console.log('buildId: '+buildId);

  const keyRingName=process.env.KEY_RING_NAME;
  const keyName=process.env.KEY_NAME;
  const githubToken=process.env.ENCRYPTED_GITHUB_TOKEN;
  const gcloudProject=process.env.GCP_PROJECT

  var states = new Map();
  states.set('QUEUED','pending');
  states.set('WORKING','pending');
  states.set('SUCCESS','success');
  states.set('CANCELLED','error');
  states.set('TIMEOUT','error');
  states.set('INTERNAL_ERROR','error');
  states.set('FAILURE','failure');

  var context='google-cloud-build/';
  if(jobType=='build+test'){
    context+='build+test/'+flavor;
  }else if(jobType=='performance-test'){
    context+='performance-test/'+flavor;
  }else if(jobType=='release'){
    context+='release';
  }else{
    console.error("Flavor not defined")
  }

  url='https://console.cloud.google.com/cloud-build/builds/'+buildId+'?project='+gcloudProject

  githubStatus = {
      state: states.get(status),
      context: context,
      description: context,
      sha: commitSha,
      token: "",
      repo: githubRepo,
      owner: githubUser,
      url: url
  };
  
  const kms = require('@google-cloud/kms');
  const client = new kms.KeyManagementServiceClient();
  const locationId = 'global';
  const name = client.cryptoKeyPath(
    gcloudProject,
    locationId,
    keyRingName,
    keyName
  );

  const ciphertext = githubToken;//Buffer.from(githubToken, 'base64').toString();
  client.decrypt({name, ciphertext})
      .then(responses => {
        const response = responses[0];
        const commitStatus = require('commit-status');
      	console.log("Send status");
      	console.log(githubStatus);
        githubStatus.token=response.plaintext;
        commitStatus.post(githubStatus);
      	console.log("Sent status");
      })
      .catch(err => {
        console.error(err);
      });
};

