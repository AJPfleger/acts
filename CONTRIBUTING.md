# Contributing to ACTS

Contributions to the ACTS project are very welcome and feedback on the documentation is greatly appreciated. In order to be able to contribute to the ACTS project, developers must have a valid CERN user account. Unfortunately, lightweight CERN accounts for external users do not have sufficient permissions to access certain CERN services used by the ACTS project.

1. [Mailing lists](#mailing-lists)
2. [Bug reports and feature requests](#bug-reports-and-feature-requests)
3. [Make a contribution](#make-a-contribution)
    1. [Setting up your fork](#setting-up-your-fork)
    2. [Creating a merge request](#creating-a-merge-request)
    3. [Workflow recommendations](#workflow-recommendations)
    4. [Coding style and guidelines](#coding-style-and-guidelines)
    5. [git tips and tricks](#git-tips-and-tricks)
4. [Administrator's corner](#admin-corner)
    1. [Setting up a Jenkins CI server](#setup-jenkins)

## <a name="mailing-lists">Mailing lists</a>

1. [acts-users@cern.ch](https://e-groups.cern.ch/e-groups/Egroup.do?egroupName=acts-users): Users of the ACTS project should subscribe to this list as it provides:
    - regular updates on the software,
    - access to the ACTS JIRA project for bug fixes/feature requests,
    - a common place for asking any kind of questions.
1. [acts-developers@cern.ch](https://e-groups.cern.ch/e-groups/Egroup.do?egroupName=acts-developers): Developers are encouraged to also subscribe to this list as it provides you with:
    - a developer role in the ACTS JIRA project (allows you to handle tickets),
    - information about developer meetings,
    - a common place for technical discussions.

## <a name="bug-reports-and-feature-requests">Bug reports and feature requests</a>

If you want to report or start a feature request, please open a ticket in the [ACTS JIRA](https://its.cern.ch/jira/projects/ACTS/) (**Note:** access is restricted to members of the mailing lists mentioned above). A comprehensive explanation will help the development team to respond in a timely manner. Therefore, the following details should be mentioned:

- bug reports
    - issue type: "Bug"
    - summary: short description of the problem
    - priority: will be set by the development team
    - components: if known, part of ACTS affected by this bug; leave empty otherwise
    - affects version: version of ACTS affected by this bug
    - a detailed description of the bug including a receipe on how to reproduce it and any hints which may help diagnosing the problem 
- feature requests
    - issue type: "Improvement" or "New Feature"
    - summary: short description of new feature
    - priority: will be set by the development team
    - a detailed description of the feature request including possible use cases and benefits for other users

## <a name="make-a-contribution">Make a contribution</a>

The instructions below should help you getting started with development process in the ACTS project. If you have any questions, feel free to ask [acts-developers@cern](mailto:acts-developers@cern.ch) for help or guidance.

### <a name="setting-up-your-fork">Setting up your fork</a>

The ACTS project uses a git repository which is hosted on the CERN GitLab server. In order to be able to create merge requests (and thus, contribute to the development of ACTS), you need to create a fork on the CERN GitLab server. A general introduction to the GitLab web interface can be found [here](https://gitlab.cern.ch/help/gitlab-basics/README.md). Very nice tutorials as well as explanations for concepts and workflows with git can be found on [Atlassian](https://www.atlassian.com/git/). For a shorter introduction and the full git documentation have a look at the [git tutorial](https://git-scm.com/docs/gittutorial).

#### Configuring git

Commits to repositories on the CERN GitLab server are only accepted from CERN users. Therefore, it is important that git is correctly configured on all machines you are working on. It is necessary to that the git user email address to the primary email address of your CERN account (usually: firstname.lastname@cern.ch). You can check the current values with:

    git config user.name
    git config user.email

You can change those settings by either editing the `.gitconfig` file in your home directory or by running:

    git config --global user.name "Donald Duck"
    git config --global user.email "donald.duck@cern.ch"
    
Further recommended settings are:

    git config --global push.default simple
    git config --global pill.rebase true

#### Creating your fork

As a first step, you need to create your own fork of the ACTS project. For doing this, please go to the [ACTS GitLab page](https://gitlab.cern.ch/acts/a-common-tracking-sw), click on the fork button, and follow the instructions ([GitLab Help "How to fork a project"](https://gitlab.cern.ch/help/gitlab-basics/fork-project.md)).

#### Configuring your fork

**Important:** Due to some limitations in the GitLab JIRA integration, you need to fix the JIRA settings for your forked project. This can be done by starting from the GitLab project page of your fork and then going to "Settings -> Services -> JIRA". Remove anything in the field "Username" and leave it empty. If you fail to do so, the ACTS JIRA project will be spammed with hundreds of duplicates comments and you will likely receive an angry email from the development team ;-).

Once you have created your fork on the CERN GitLab server, you need to create a local copy to start coding. This is done by cloning your forked repository onto your local machine through the following command (if you are on a UNIX system):

    git clone <FORK_URL> <Destination>

- &lt;FORK_URL&gt; is the web address of your forked repository which can be found on the project page in the GitLab web interface
- &lt;DESTINATION&gt; is optional and gives the location on your local machine where the clone will be created

You probably want to be able to pull in changes from the official ACTS repository to benefit from the latest and greatest improvements. This requires that you add the official ACTS repository as another remote to your local clone. 

    cd <DESTINATION>
    git remote add ACTS ssh://git@gitlab.cern.ch:7999/acts/a-common-tracking-sw.git

You can check that everything went ok with

    git remote -v

where the reference to the ACTS repository should appear (along with your forked repository on the CERN GitLab server). This procedure is also described on [github](https://help.github.com/articles/configuring-a-remote-for-a-fork/).

#### Keeping your fork up-to-date

At certain points you may want to sync your fork with the latest updates from the official ACTS repository. The following commands illustrate how to update the 'master' branch of fork. The same procedure can be used to sync any other branch, but you will rarely need this. Please mak sure to commit/stash all changes before proceeding to avoid any loss of data. The following commands must be run in the working directory of the local clone or your forked repository. 

    git fetch ACTS
    git checkout master
    git merge --ff-only ACTS/master
    git push origin master

### <a name="creating-a-merge-request">Creating a merge request</a>

Once your development is ready for integration, you should open a merge request at the [ACTS project](https://gitlab.cern.ch/acts/a-common-tracking-sw) ([GitLab Help: Create a merge request](https://gitlab.cern.ch/help/gitlab-basics/add-merge-request.md)). The target branch should usually be _master_ for feature requests and _releas-X,Y,Z_ for bugfixes. The ACTS projects accepts only fast-foward merges which means that your branch must have been rebased on the target branch. This can be achieved by updating your fork as described above and then run:

    git checkout <my_feature_branch>
    git rebase -i origin/<target_branch>
    git push
    
At this point you should make use of the interactive rebase procedure to clean up your commits (squash small fixes, improve commit messages etc; [Rewriting history](https://robots.thoughtbot.com/git-interactive-rebase-squash-amend-rewriting-history)).  
Merge requests are required to close a ACTS JIRA ticket. This is achieved by adding e.g. 'fixes ACTS-XYZ' to the end of the merge request description. Please note that JIRA tickets should only be referenced by merge requests and not individual commits (since strictly there should only be one JIRA ticket per merge request). Once the merge request is opened, a continous integration job is triggered which will add multiple comments to the merge request (e.g. build status, missing license statements, doxygen errors, test coverage etc). Please have a look at them and fix them by adding more commits to your branch if needed.  
Please find below a short checklist for merge requests.

#### Checklist for merge requests

- Your branch has been rebased on the target branch and can be integrated through a fast-forward merge.
- A detailed description of the merge request is provided which includes a reference to a JIRA ticket (e.g. `Closes ACTS-1234`, [GitLab Help: Closing JIRA tickets](http://docs.gitlab.com/ee/project_services/jira.html#closing-jira-issues)).
- All files start with the MPLv2 license statement.
- All newly introduced functions and classes have been documented properly with doxygen.
- Unit tests are provided for new functionalities.
- For bugfixes: a test case has been added to avoid the re-appearance of this bug in the future.
- All added cmake options were added to 'cmake/PrintOptions.cmake'.

### <a name="workflow-recommendations">Workflow recommendations</a>

In the following a few recommendations are outlined which should help you to get familiar with development process in the ACTS project.

1. **Each development its own branch!**  
Branching in git is simple, it is fun and it helps you keep your working copy clean. Therefore, you should start a new branch for every development. All work which is logically/conceptually linked should happen in one branch. Keep your branches short. This helps immensly to understand the git history if you need to look at it in the future.
If projects are complex (e.g. large code refactoring or complex new features), you may want to use _sub_-branches from the main development branch as illustrated in the picture below.  

<img src="doc/figures/sub_dev.png" alt="workflow for large feature">
1. **Never, ever directly work on any "official" branch!**  
Though not strictly necessary and in the end it is up to you, it is strongly recommended that you never commit directly on a branch which tracks an "official" branch. As all branches are equal in git, the definition of "official" branch is quite subjective. In the ACTS project you should not work directly on branches which are **protected** in the CERN GitLab repository. Usually, these are the _master_ and _release-X.Y.Z_ branches. The benefit of this strategy is that you will never have problems to update your fork. Any git merge in your local repository on such an "official" branch will always be a fast-forward merge.

1. **Use atomic commits!**  
Similarly to the concept of branches, each commit should reflect a self-contained change. Try to avoid overly large commits (bad examples are for instance mixing logical change with code cleanup and typo fixes).

1. **Write good commit messages!**  
Well-written commit messages are key to understand your changes. There are many guidelines available on how to write proper commit logs (e.g. [here](http://alistapart.com/article/the-art-of-the-commit), [here](http://chris.beams.io/posts/git-commit/), or [here](https://wiki.openstack.org/wiki/GitCommitMessages#Information_in_commit_messages)). As a short summary:
    - Structure your commit messages into short title (max 50 characters) and longer description (max width 72 characters)!
      This is best achieved by avoiding the `commit -m` option. Instead write the commit message in an editor/git tool/IDE... 
    - Describe why you did the change (git diff already tells you what has changed)!
    - Mention any side effects/implications/consquences!

1. **Prefer git pull --rebase!**  
If you work with a colleague on a new development, you may want to include his latest changes. This is usually done by calling `git pull` which will synchronise your local working copy with the remote repository (which may have been updated by your colleague). By default, this action creates a merge commit if you have local commits which were not yet published to the remote repository. These merge commits are considered to contribute little information to the development process of the feature and they clutter the history (read more e.g.  [here](https://developer.atlassian.com/blog/2016/04/stop-foxtrots-now/) or [here](http://victorlin.me/posts/2013/09/30/keep-a-readable-git-history)). This problem can be avoided by using `git pull --rebase` which replays your local (un-pushed) commits on the tip of the remote branch. You can make this the default behaviour by running `git config pull.rebase true`. More about merging vs rebasing can be found [here](https://www.atlassian.com/git/tutorials/merging-vs-rebasing/).
1. **Push your development branches as late as possible!**  
Unless required by other circumstances (e.g. collaboration with colleagues, code reviews etc) it is recommended to push your development branch once you are finished. This gives you more flexibility on what you can do with your local commits (e.g. rebase interactively) without affecting others. Thus, it minimises the risk for running into git rebase problems.
1. **Update the documentation!**  
Make sure that the documentation is still valid after your changes. Perform updates where needed and ensure integrity between the code and its documentation. 

### <a name="coding-style-and-guidelines">Coding style and guidelines</a>

The ACTS project uses [clang-format](http://clang.llvm.org/docs/ClangFormat.html) (currently v3.8.0) for formatting its source code. A `.clang-format` configuration file comes with the project and should be used to automatically format the code. There are several instructions available on how to integrate clang-format with your favourite IDE (e.g. [eclipse](https://marketplace.eclipse.org/content/cppstyle), [Xcode](https://github.com/travisjeffery/ClangFormat-Xcode), [emacs](http://clang.llvm.org/docs/ClangFormat.html#emacs-integration)). The ACTS CI system will automatically apply code reformatting using the provided clang-format configuration once merge requests are opened. However, developers are strongly encouraged to use this code formatter also locally to reduce conflicts due to formatting issues.

In addition, the following conventions are used in ACTS code:

- Class names start with a capital letter.
- Function names start with a lower-case letter and use camel-case.
- Names of class member variables start with `m_`.
- getter methods are called like the corresponding member variable without the prefix 'get' (e.g. `covariance()` instead of `getCovariance()`)
- setter methods use the prefix 'set' (e.g. `setCovariance(...)`)
- passing arguments to functions:
    - by value for simple data types (e.g. int, float double, bool)
    - by constant reference for required input of non-trivial type
    - by (raw) pointer for optional input of non-trivial type
    - only use smart pointers if the function called must handle ownership (very few functions actually do)
- returning results from functions:
    - newly created objects should be returned<br />
      a) as unique pointer if the object is of polymorphic type or its presence is not always ensured<br />
      b) by value if the object is of non-polymorphic type and always exists
    - existing objects (e.g. member variables) should be returned by<br />
      a) const reference for custom types with costly copy constructors<br />
      b) value in all other cases   
- Doxygen documentation:
    - Put all documentation in the header files. 
    - Use `///` as block comment (instead of `/* ... */`).
    - Doxygen documentation goes in front of the documented entity (class, function, (member) variable).
    - Use the \@<cmd> notation.
    - Functions and classes must have the \@brief description.
    - Document all (template) parameters using \@(t)param and explain the return value for non-void functions. Mention important conditions which may affect the return value.
    - Use `@remark` to specify pre-conditions.
    - Use `@note` to provide additional information.
    - Link other related entities (e.g. functions) using `@sa`. 
 
### <a name="git-tips-and-tricks">git tips and tricks</a>

The following section gives some advise on how to solve certain situations you may encounter during your development process. Many of these commands have the potential to loose uncommitted data. So please make sure that you understand what you are doing before running the receipes below. Also, this collection is non-exhaustive and alternative approaches exist. If you want to contribute to this list, please drop an email to [acts-developers@cern.ch](mailto:acts-developers@cern.ch).

**Before doing anything**  
In the rare event that you end up in a situation you do not know how to solve, get to a clean state of working copy and create a (backup) branch, then switch back to the original branch. If anything goes wrong, you can always checkout the backup branch and you are back to where you started.  
**Modify the author of a commit**  
If your git client is not correctly set up on the machine you are working on, it may derive the committer name and email address from login and hostname information. In this case your commits are likely rejected by the CERN GitLab server. As a first step, you should correctly configure git on this machine as described above so that this problems does not appear again.  
a)  You are lucky and only need to fix the author of the latest commit. You can use `git commit --amend`:

    git commit --amend --no-edit --author "My Name <login@cern.ch>
    
b) You need to fix (several) commit(s) which are not the current head. You can use `git rebase`:  
For the following it is assumed that all commits which need to be fixed are in the same branch &lt;BRANCH&gt;, and &lt;SHA&gt; is the hash of the earliest commit which needs to be corrected.

    git checkout <BRANCH>
    git rebase -i -p <SHA>^
    
In the editor opened by the git rebase procedure, add the following line after each commit you want to fix:

    exec git commit --amend --author="New Author Name <email@address.com>" -C HEAD
    
Then continue with the usual rebase procedure.

**Make a bugfix while working on a feature**  
    During the development of a new feature you discover a bug which needs to be fixed. In order to not mix bugfix and feature development, the bugfix should happen in a different branch. The recommended procedure for handling this situation is the following:
1. Get into a clean state of your working directory on your feature branche (either by commiting open changes or by stashing them).
1. Checkout the branch the bugfix should be merged into (either _master_ or _release-X.Y.Z_) and get the most recent version.
1. Create a new branch for the bugfix.
1. Fix the bug, write a test, update documentation etc.
1. Open a merge request for the bug fix.
1. Switch back to your feature branch.
1. Merge your local bugfix branch into the feature branch. Continue your feature development.
1. Eventually, the bugfix will be merged into _master_. Then, you can rebase your feature branch on master which will remove all duplicate commits related to the bugfix.    

In git commands this looks like:

    git stash
    git checkout master && git pull
    git checkout -b <bug_fix_branch>
    # Implement the bug fix, add tests, commit.
    # Open a merge request.
    git checkout <feature_branch>
    git merge <bug_fix_branch>
    # Once the merge request for the bug fix is accepted in the upstream repository:
    git fetch
    git rebase origin/master

This should give the following git history where the initial feature branch is blue, the bugfix branch is yellow and the feature branch after the rebase is red.
<img src="doc/figures/bugfix_while_on_feature.png" alt="fixing a bug while working on a feature">     
**Move most recent commit(s) to new branch**  
Very enthusiastic about the cool feature you are going to implement, you started from master and made one (or more) commits for your development. That's when you realised that you did not create a new branch and committed directly to the master branch. As you know that you should not directly commit to any "official" branch, you are desperately looking for a solution. Assuming your current situation is: A -> B -> C -> D -> E where HEAD is pointing to E (= master) and the last "official" commit is B as shown below.
<img src="doc/figures/move_to_branch1.png" alt="moving commits to new branch">
You can resolve this situation by running:

    git checkout <new_branch_name>
    git reset --hard <hash of B>
    git checkout <new_branch_name>
    
which should give the following situation:  
<img src="doc/figures/move_to_branch2.png" alt="moving commits to new branch">  
Now, master is pointing to B, HEAD and &lt;new\_branch\_name&gt; are pointing to E and you can happily continue with your work.

## <a name="admin-corner">Administrator's corner</a>

This section gives useful information to the administrators of the ACTS project. For normal developers the sections below are irrelevant.

### <a name="setup-jenkins">Setting up a Jenkins CI server</a>

The following steps explain on how to setup and configure a Jenkins server for continuous integration tests using the CERN openstack infrastructure.

1. Launch an openstack instance as described [here](https://clouddocs.web.cern.ch/clouddocs/tutorial_using_a_browser/index.html). The following settings are recommended:
  + flavor: m2.medium
  + boot from image: Ubuntu 16.04 LTS - x86_64
  + keypair: make sure to add a public ssh key using RSA encryption (**Note**: A DSA encrypted key does not work anymore with Ubuntu 16.04).
2. Login to your virtual machine, update the system and setup a user with root privileges:

        # login to the machine using the key-pair provided during instance creation
        ssh -i <public key> ubuntu@<vm-name>
        # update system 
        sudo apt-get update 
        sudo apt-get upgrade
        # fix locale warnings
        sudo locale-gen "en_GB.UTF-8"
        sudo dpkg-reconfigure locales
        # add ACTS jenkins user with sudo privileges
        sudo adduser atsjenkins
        sudo usermod -aG sudo atsjenkins
        # to enable password authentication:
        # change PasswordAuthentication to 'yes' in '/etc/ssh/sshd_config'
        sudo service ssh restart
        # generate public-private key pair
        ssh-keygen -t rsa
        # add the public key from '~/.ssh/id_rsa.pub' to the GitLab atsjenkins account (under Profile Settings -> SSH keys)

3. Install required software:

        # compilers
        sudo apt-get install g++
        sudo apt-get install clang
        sudo apt-get install clang-format
        # cmake
        sudo apt-get install cmake
        # doxygen
        sudo apt-get install doxygen
        sudo apt-get install graphviz        
        # Eigen algebra library
        wget http://bitbucket.org/eigen/eigen/get/3.2.9.tar.gz
        tar xf 3.2.9.tar.gz
        sudo mkdir -p /opt/eigen/
        sudo mv eigen-eigen-dc6cfdf9bcec/ /opt/eigen/3.2.9
        rm 3.2.9.tar.gz
        # Boost library
        wget -O boost-1_61_0.tar.gz http://downloads.sourceforge.net/project/boost/boost/1.61.0/boost_1_61_0.tar.gz?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fboost%2Ffiles%2Fboost%2F1.61.0%2F&ts=1474572837&use_mirror=freefr
        tar xf boost-1_61_0.tar.gz
        cd boost_1_61_0/
        ./bootstrap.sh --prefix=/opt/boost/1.61.0
        sudo ./b2 install
        cd .. && rm boost-1_61_0.tar.gz && rm -r boost_1_61_0
        # ROOT
        wget https://root.cern.ch/download/root_v6.06.08.source.tar.gz
        tar xf root_v6.06.08.source.tar.gz
        mkdir build && cd build
        sudo apt-get install -y libx11-dev libxpm-dev libxft-dev libxext-dev libfftw3-dev libxml2-dev libgsl-dev
        cmake ../root-6.06.08/ -DCMAKE_INSTALL_PREFIX=/opt/root/6.06.08 -Dcxx14=ON -Dminuit2=ON -Droofit=ON -Dxml=ON -Dfftw3=ON -Dgdml=ON -Dopengl=ON -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
        sudo cmake --build . --target install -- -j 4
        cd .. && rm -rf build/ && rm -rf root-6.06.08/ && rm root_v6.06.08.source.tar.gz
        # Python stuff
        sudo apt install python-pip
        sudo pip install requests
        
4. Install Jenkins (taken from [here](https://wiki.jenkins-ci.org/display/JENKINS/Installing+Jenkins+on+Ubuntu)):

        wget -q -O - https://pkg.jenkins.io/debian/jenkins-ci.org.key | sudo apt-key add -
        sudo sh -c 'echo deb http://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
        sudo apt-get update
        sudo apt-get install jenkins
        sudo service jenkins start
        
5. Setup a convenience alias and SSH credentials:

        # create an alias for executing commands as jenkins user
        cat >> ~/.bash_aliases << EOF
        > alias asjenkins="sudo -u jenkins"
        > EOF
        source ~/.bash_aliases
        # copy atsjenkins SSH credentials
        sudo cp ~/.ssh/id_rsa ~/.ssh/id_rsa.pub /var/lib/jenkins/.ssh
        sudo chown jenkins /var/lib/jenkins/.ssh/id_rsa /var/lib/jenkins/.ssh/id_rsa.pub
        sudo chgrp jenkins /var/lib/jenkins/.ssh/id_rsa /var/lib/jenkins/.ssh/id_rsa.pub

6. Open firewall ports:

        # install firewall configuration tool
        sudo apt-get install firewalld
        sudo firewall-cmd --zone=public --add-port=8080/tcp --permanent
        sudo firewall-cmd --zone=public --add-service=http --permanent
        sudo firewall-cmd --reload

7. Configure the Jenkins instance through the web interface:
    1. Open the Jenkins Dashboard under http://<VM-name>:8080 in your browser and follow the instructions to unlock the Jenkins instance.
    2. Install the following Jenkins plugins:
        + git
        + gitlab
        + multijob
        + embeddable build status
        + rebuilder
        + timestamper
        + conditional build step
        + parametrized trigger
        + workspace cleanup
        + environment script
    3. Create a Jenkins admin user with name `atsjenkins`, select a password and use `ats.jenkins@cern.ch` as email.
    4. Configure Jenkins instance:
        + Manage Jenkins -> Configure Global Security: enable "Allow anonymous read access"
        + Manage Jenkins -> Configure System 
            + Maven Project Configuration:
                + # executors: 5
            + GitLab section:
                + Connection name: GitLab
                + GitLab host URL: https://gitlab.cern.ch
                + add credentials: use type "GitLab API token" and insert the private token from GitLab atsjenkins user (which can be found in GitLab under Profile settings -> Account -> Private token)
                + under advanced: tick "Ignore SSL certificate errors"
                + hit "Test Connection" which should return "success"
            + Jenkins location:
                + URL: http://<VM-name>.cern.ch:8080
                + email: ats.jenkins@cern.ch
            + Git plugin:
                + user.name: ATS Jenkins
                + user.email: ats.jenkins@cern.ch
8. Configure the Jenkins CI jobs:

        # checkout the job configuration and helper scripts
        cd /var/lib/jenkins
        asjenkins git init
        asjenkins git remote add origin ssh://git@gitlab.cern.ch:7999/acts/jenkins-setup.git
        asjenkins git fetch
        asjenkins git checkout -t origin/master
        
        # fixing some credential settings which means manually copying the following part from 'credentials.xml.in'
        # <com.cloudbees.jenkins.plugins.sshcredentials.impl.BasicSSHUserPrivateKey ...>
        # ...
        # </com.cloudbees.jenkins.plugins.sshcredentials.impl.BasicSSHUserPrivateKey>
        # into 'credentials.xml' after the following line:
        # <java.util.concurrent.CopyOnWriteArrayList>
        
        # allow passing undefined parameters:
        # add '-Dhudson.model.ParametersAction.keepUndefinedParameters=true' to JAVA_ARGS in /etc/default/jenkins
        
        # set some symlinks
        sudo sh create_symlinks.sh
        
        # restart the jenkins server
        sudo service jenkins restart
            
9. Setup kerberos authentication needed for updating the ACTS webpage with new tags

        # authentication
        sudo apt install krb5-user
        sudo apt-get install openafs-client
        # generate keytab file
        ktutil
        ktutil:  addent -password -p atsjenkins@CERN.CH -k 1 -e aes256-cts-hmac-sha1-96
        ktutil:  addent -password -p atsjenkins@CERN.CH -k 1 -e arcfour-hmac
        ktutil:  wkt .keytab
        ktutil:  q
        sudo mv .keytab /etc/krb5_atsjenkins.keytab
        sudo chown jenkins /etc/krb5_atsjenkins.keytab
        sudo chgrp jenkins /etc/krb5_atsjenkins.keytab
        # download kerberos configuration
        sudo wget -O /etc/krb5.conf http://linux.web.cern.ch/linux/docs/krb5.conf
        # add "@daily ID=afstoken kinit --renew" to /etc/crontab
        
        # add the following to /etc/ssh/ssh_config
        #
        # HOST lxplus*
        #   ForwardX11 yes
        #   ForwardX11Trusted no
        #   GSSAPITrustDNS yes
        #   HashKnownHosts yes
        #   GSSAPIAuthentication yes
        #   GSSAPIDelegateCredentials yes
        
        # install mailutils for sending notifications
        sudo apt-get install mailutils
        
/// @ingroup Contributing