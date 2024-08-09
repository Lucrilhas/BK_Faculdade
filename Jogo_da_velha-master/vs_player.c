// Jogo da velha criado por Lucas Elias de Andrade Cruvinel
// Email : LucrilhasBR@hotmail.com ou LucrilhasBR@gmail.com
// Obs : Jogo feito para windowns, nao ira funcionar como deveria em outras sistemas

// A seguir estão as bibliotecas uadas:
#include <stdlib.h>
#include <stdio.h>
#include <locale.h>
#include <windows.h>
#include<time.h>

//Este arquivo tem a principal função de ser o jogo PVP (Player vs Player)

void verificador_pvp(char velha[3][3], char nomep1[10], char nomep2[10], int  round){ // Função que acontece após cada rodada para checar se alguem canhou

    // A seguir estão as condicões, dependendo o jogo acaba e é redirecionado para uma tela de parabens ou de empate
    if          (velha[0][0]=='X' && velha[0][1]=='X' && velha[0][2] == 'X'){parabeins_X(velha, nomep1, nomep2);}
    else if(velha[1][0]=='X' && velha[1][1]=='X' && velha[1][2] == 'X'){parabeins_X(velha, nomep1, nomep2);}
    else if(velha[2][0]=='X' && velha[2][1]=='X' && velha[2][2] == 'X'){parabeins_X(velha, nomep1, nomep2);}

    else if(velha[0][0]=='X' && velha[1][0]=='X' && velha[2][0] == 'X'){parabeins_X(velha, nomep1, nomep2);}
    else if(velha[0][1]=='X' && velha[1][1]=='X' && velha[2][1] == 'X'){parabeins_X(velha, nomep1, nomep2);}
    else if(velha[0][2]=='X' && velha[1][2]=='X' && velha[2][2] == 'X'){parabeins_X(velha, nomep1, nomep2);}

    else if(velha[0][0]=='X' && velha[1][1]=='X' && velha[2][2] == 'X'){parabeins_X(velha, nomep1, nomep2);}
    else if(velha[2][0]=='X' && velha[1][1]=='X' && velha[0][2] == 'X'){parabeins_X(velha, nomep1, nomep2);}

    else if(velha[0][0]=='O' && velha[0][1]=='O' && velha[0][2] == 'O'){parabeins_O(velha, nomep1, nomep2);}
    else if(velha[1][0]=='O' && velha[1][1]=='O' && velha[1][2] == 'O'){parabeins_O(velha, nomep1, nomep2);}
    else if(velha[2][0]=='O' && velha[2][1]=='O' && velha[2][2] == 'O'){parabeins_O(velha, nomep1, nomep2);}

    else if(velha[0][0]=='O' && velha[1][0]=='O' && velha[2][0] == 'O'){parabeins_O(velha, nomep1, nomep2);}
    else if(velha[0][1]=='O' && velha[1][1]=='O' && velha[2][1] == 'O'){parabeins_O(velha, nomep1, nomep2);}
    else if(velha[0][2]=='O' && velha[1][2]=='O' && velha[2][2] == 'O'){parabeins_O(velha, nomep1, nomep2);}

    else if(velha[0][0]=='O' && velha[1][1]=='O' && velha[2][2] == 'O'){parabeins_O(velha, nomep1, nomep2);}
    else if(velha[2][0]=='O' && velha[1][1]=='O' && velha[0][2] == 'O'){parabeins_O(velha, nomep1, nomep2);}

    else if(round == 9){empate(velha, nomep1, nomep2);}

}

void jog_vs_jog(){ // Função que começa o jogo pvp, declarando as variaveis necessarias para o jogo e as configurando
    system("cls"); setlocale(LC_ALL, "Portuguese");

    int i, j, k, numround = 1;  //Declara os inteiros
    char nomep1[10], nomep2[10]; // Declara a string que será usada para armazenar o nome dos jogadores
    char velha[3][3], amostrador[3][3];; // Declara as matrizes que serão usadas como tabuleiro

 	for(i=0, k=49;i<3;i++){for(j=0;j<3;j++){velha[i][j] = NULL;amostrador[i][j] = k;k++;}} //Colocando NULL na matriz na velha, e Botando numeros de 1 a 9 no amostrador

 	printf("\n > Nome do player 1 (Será o X):  "); //Pegando nome dos players
 	scanf("%s", &nomep1);
 	printf("\n > Nome do player 2 (Será o O):  ");
 	scanf("%s", &nomep2);

 	rounds(velha, amostrador, nomep1, nomep2, numround); // Função em que realmente começa o jogo

}

void rounds(char velha[3][3], char amostrador[3][3], char nomep1[10], char nomep2[10], int qual_round){// Funçãoque realmente aconte o jogo, funciona como uma função recursiva que vai se chamando porém cada vez com um numero de round diferente
        system("cls"); setlocale(LC_ALL, "Portuguese");

        if(qual_round == 1 || qual_round == 3 || qual_round == 5 || qual_round == 7 || qual_round == 9)     {system("color 0A");} //Caso seja vez do X a cor será essa
        if(qual_round == 2 || qual_round == 4 || qual_round == 6 || qual_round == 8)                                        {system("color 0C");} //Caso seja vez do O a cor será essa

        int linha, coluna; //Declarando variaveis auxiliares
        char escolha; // Variavel que pega a escolha dos jogadores

        imprime_velha(velha, amostrador); //função que mostra o tableiro

        printf("||                                                    ||\n");

        //Condições para modificar o layout de acordo com de quem é a vez
        if(qual_round == 1 || qual_round == 3 || qual_round == 5 || qual_round == 7 || qual_round == 9)   {printf("||                Vez do X:                           ||\n");}
        if(qual_round == 2 || qual_round == 4 || qual_round == 6 || qual_round == 8)                                      {printf("||                Vez do O:                           ||\n");}

        if(qual_round == 1 || qual_round == 3 || qual_round == 5 || qual_round == 7 || qual_round == 9)   {printf("                    > > %s < <\n", nomep1);}
        if(qual_round == 2 || qual_round == 4 || qual_round == 6 || qual_round == 8)                                      {printf("                    > > %s < <\n", nomep2);}

        printf("|| Escolha um número de 1 até 9 correspondente        ||\n");
        printf("|| a posição desejada no tabuleiro.                   ||\n");
        printf("||                                                    ||\n");
        printf("||====================================================||\n");

        escolha = getch(); // Pega a escolha do jogador

        //Condições que transformam as escolhas em cordenadas na matriz
                         if (escolha == '1'){linha = 0; coluna = 0;}
            else    if (escolha == '2'){linha = 0; coluna = 1;}
            else    if(escolha == '3'){linha = 0; coluna = 2;}
            else    if(escolha == '4'){linha = 1; coluna = 0;}
            else    if(escolha == '5'){linha = 1; coluna = 1;}
            else    if(escolha == '6'){linha = 1; coluna = 2;}
            else    if(escolha == '7'){linha = 2; coluna = 0;}
            else    if(escolha == '8'){linha = 2; coluna = 1;}
            else    if(escolha == '9'){linha = 2; coluna = 2;}

         // Condição que verifica se a a posição é válida
            if(velha[linha][coluna] == 'X' || velha[linha][coluna] == 'O'){ // Se não for válida ele chama a funçao round com o mesmo numéro de round
                printf("\n > > POSIÇÃO INVALIDA!! < <\n");
                system("pause");
                rounds(velha, amostrador, nomep1, nomep2, qual_round);}

           else{ // Caso sejá válido ela escreve na matriz, aumenta o número de round e vai para o proximo

                if(qual_round == 1 || qual_round == 3 || qual_round == 5 || qual_round == 7 || qual_round == 9){velha[linha][coluna] = 'X'; amostrador[linha][coluna] = NULL;}
                if(qual_round == 2 || qual_round == 4 || qual_round == 6 || qual_round == 8){velha[linha][coluna] = 'O';amostrador[linha][coluna] = NULL;}

                verificador_pvp(velha, nomep1, nomep2, qual_round); //verifica se alguem vence
                qual_round++;
                rounds(velha, amostrador, nomep1, nomep2,qual_round);
                }
}
