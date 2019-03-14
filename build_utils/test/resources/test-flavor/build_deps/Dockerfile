FROM {{language_deps}}


RUN apt-get -y update && \
    apt-get install -y curl openjdk-8-jdk && \
	echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
	curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
	apt-get update && \
	apt-get install -y bazel && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig