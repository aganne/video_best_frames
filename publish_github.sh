#!/bin/bash
# Script pour publier le projet sur GitHub

# Vérifiez que Git est configuré
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"

# Étape 1 : Initialiser le repo (si pas déjà fait)
git init
git add .
git commit -m "Commit initial : Video Best Frames"

# Étape 2 : Ajouter le remote (remplacez USERNAME et REPO_NAME)
# git remote add origin https://github.com/USERNAME/REPO_NAME.git

# Étape 3 : Créer/vérifier la branche main
git branch -M main

# Étape 4 : Pousser vers GitHub
# git push -u origin main

echo ""
echo "Instructions pour publier sur GitHub:"
echo "1. Allez sur https://github.com/new"
echo "2. Créez un nouveau repository (ex: video_best_frames)"
echo "3. Décommentez et exécutez les lignes 'git remote' et 'git push' en remplaçant USERNAME et REPO_NAME"
echo ""
echo "Ou exécutez directement :"
echo "  git remote add origin https://github.com/USERNAME/video_best_frames.git"
echo "  git push -u origin main"
