# Instructions for agents

Welcome. This project aims to find & implement a recent pattern recognition 
algorithm.

## BEFORE YOU START…

Please read the full handout on [handout.md](handouts-do-not-upload/handout.md) 
before you start to work. This includes the detailed information on what you're 
supposed to work on.

DO NOT copy the handout out elsewhere, or otherwise commit it to the repository 
in any way. This file is copyrighted material of my university, and is not to 
be shared online.

## Environment setup

When you run commands, use zsh. FYI the `~/.zprofile` includes environment 
variables that you will use. 

### Python

Python in this project is managed by Conda. A Conda environment is available at 
`/opt/anaconda3-pbe/envs/ptrc-hw03` (name `ptrc-hw03`). Therefore, venvs are 
not required.

`ptrc-hw03` includes the essential packages (via the Anaconda Distribution) and 
uses Python 3.14.10. **DO NOT install older versions of Python.** PyTorch and 
many dependencies supports Python 3.14.  
If you have a reason to install an older version of Python, do research on it, 
and only then raise it up to my attention for approval.

### Git

This is a brand-new Git repository. There are no remote repositories because 
one hasn't been set up yet. You are expected to commit your work regularly.

When you commit your work, use the command line. There are two things you 
should do for each commit:

  1. Add `--author $PTRC_GENAI_ATTR`. This attributes the author of the commit 
     to you, but leaves the committer as me so I can sign the commit. Do not 
     modify any Git configs.
  2. Append `Co-authored-by: $PTRC_USER_ATTR` to the commit message. This makes 
     my name show up on the GitHub interface as a co-author.

A good commit should look like this:

```
% git cat-file commit HEAD
tree [hash]
author [$PTRC_GENAI_ATTR] [timestamp]    # set with `git commit --author`
committer [$PTRC_USER_ATTR] [timestamp]  # alreday set with Git config
gpgsig -----BEGIN PGP SIGNATURE-----

[PTRC_USER's signature]                  # already set with Git config

 -----END PGP SIGNATURE-----

[>>> YOUR MESSAGE HERE! <<<]

Co-authored-by: [$PTRC_USER_ATTR]        # set by your commit message
```

## Working on the project

You should log [agent-logs.md](agent-logs.md) what you're doing and your 
rationale of each change you make. When you use online materials, you should 
also leave, at least, the title of the webpage/document so it can be traced.

Although the handout calls for a Word/PDF report, you should not create one. 
What you should do instead is to make your report draft in Markdown form. I 
will incorporate extra information of my own into the report before submitting.

In your code, agent-logs and report draft, do not mention the name of my 
university or the name of the course, as this repository will be made public.

If you have any questions, don't hesitate to ask me. I might not be the 
smartest person ever, but we are working together here.