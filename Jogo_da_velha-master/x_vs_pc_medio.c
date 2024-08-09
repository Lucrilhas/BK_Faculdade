// Jogo da velha criado por Lucas Elias de Andrade Cruvinel
// Email: LucrilhasBR@hotmail.com ou LucrilhasBR@gmail.com
// Obs : Jogo feito para windowns, nao ira funcionar como deveria em outras sistemas

// A seguir estão as bibliotecas uadas:
#include <stdlib.h>
#include <stdio.h>
#include <locale.h>
#include <windows.h>
#include<time.h>

//Este arquivo tem a principal função de ser o jogo do jogador como X contra CPU no medio

void x_vs_pc_rounds_medio(char velha[3][3], char amostrador[3][3], char nomeplayer[10], int qual_round){
    system("cls"); setlocale(LC_ALL, "Portuguese");
    srand(time(0)); // Seed para o randomizador

    if(qual_round == 1 || qual_round == 3 || qual_round == 5 || qual_round == 7 || qual_round == 9)     {system("color 0A");} //Muda a cor de acordo com quem é a vez
    if(qual_round == 2 || qual_round == 4 || qual_round == 6 || qual_round == 8)                                        {system("color 0C");}

    int linha, coluna;// Declara as variaveis auxiliares
    int aux, bole;
    char escolha;

     imprime_velha(velha, amostrador);// Mostra o tabuleiro

           // Adequa o layout de acordo com quem é a vez
        if(qual_round == 1 || qual_round == 3 || qual_round == 5 || qual_round == 7 || qual_round == 9)   {printf("||                Vez do X:                           ||\n");}
        if(qual_round == 2 || qual_round == 4 || qual_round == 6 || qual_round == 8)                                      {printf("||                Vez do O:                           ||\n");}

        if(qual_round == 1 || qual_round == 3 || qual_round == 5 || qual_round == 7 || qual_round == 9)   {printf("                    > > %s < <\n", nomeplayer);}
        if(qual_round == 2 || qual_round == 4 || qual_round == 6 || qual_round == 8)                                      {printf("                    > > CPU < <\n");}

        if(qual_round == 1 || qual_round == 3 || qual_round == 5 || qual_round == 7 || qual_round == 9){
            printf("|| Escolha um número de 1 até 9 correspondente        ||\n");
            printf("|| a posição desejada no tabuleiro.                   ||\n");
            printf("||                                                    ||\n");
            printf("||====================================================||\n");}

        if(qual_round == 2 || qual_round == 4 || qual_round == 6 || qual_round == 8){
            printf("||   Por favor espere:                                ||\n");
            printf("||   A CPU está pensando na melhor jogada.            ||\n");
            printf("||                                                    ||\n");
            printf("||====================================================||\n");
            Sleep(1000);}

            if(qual_round == 1 || qual_round == 3 || qual_round == 5 || qual_round == 7 || qual_round == 9){//Caso seja vez do jogador pega sua escolha

            escolha = getch(); //pega escolha do jogador

                      //Transforma a escolha em coordenadas matriz
                if (escolha == '1'){linha = 0; coluna = 0;}
                else if (escolha == '2'){linha = 0; coluna = 1;}
                else if(escolha == '3'){linha = 0; coluna = 2;}
                else  if(escolha == '4'){linha = 1; coluna = 0;}
                else  if(escolha == '5'){linha = 1; coluna = 1;}
                else   if(escolha == '6'){linha = 1; coluna = 2;}
                else  if(escolha == '7'){linha = 2; coluna = 0;}
                else  if(escolha == '8'){linha = 2; coluna = 1;}
                else if(escolha == '9'){linha = 2; coluna = 2;}

                      // Condição que verifica se a a posição é válida
            if(velha[linha][coluna] == 'X' || velha[linha][coluna] == 'O'){// Se não for válida ele chama a funçao round com o mesmo numéro de round
                printf("\n > > POSIÇÃO INVALIDA!! < <\n");
                system("pause");
                x_vs_pc_rounds_medio(velha, amostrador, nomeplayer, qual_round);}

            else{ // Caso sejá válido ela escreve na matriz, aumenta o número de round e vai para o proximo
                velha[linha][coluna] = 'X';
                amostrador[linha][coluna] = NULL;
                verificador_xvspc(velha, nomeplayer, qual_round);
                qual_round++;
                x_vs_pc_rounds_medio(velha, amostrador, nomeplayer, qual_round);}
            }

                if(qual_round == 2 || qual_round == 4 || qual_round == 6 || qual_round == 8){//Vez da CPU

                    //Condiçoes para dificultar o jogo
                    if          (velha[0][0] == 'O' && velha[0][1] == 'O' && velha[0][2] == NULL){ linha = 0; coluna = 2;}
                    else if(velha[0][0] == 'O' && velha[0][2] == 'O' && velha[0][1] == NULL){ linha = 0; coluna = 1;}
                    else if(velha[0][1] == 'O' && velha[0][2] == 'O' && velha[0][0] == NULL){ linha = 0; coluna = 0;}

                    else if(velha[1][0] == 'O' && velha[1][1] == 'O' && velha[1][2] == NULL){ linha = 1; coluna = 2;}
                    else if(velha[1][0] == 'O' && velha[1][2] == 'O' && velha[1][1] == NULL){ linha = 1; coluna = 1;}
                    else if(velha[1][1] == 'O' && velha[1][2] == 'O' && velha[1][0] == NULL){ linha = 1; coluna = 0;}

                    else if(velha[2][0] == 'O' && velha[2][1] == 'O' && velha[2][2] == NULL){ linha = 2; coluna = 2;}
                    else if(velha[2][0] == 'O' && velha[2][2] == 'O' && velha[2][1] == NULL){ linha = 2; coluna = 1;}
                    else if(velha[2][1] == 'O' && velha[2][2] == 'O' && velha[2][0] == NULL){ linha = 2; coluna = 0;}


                    else if(velha[0][0] == 'O' && velha[1][0] == 'O' && velha[2][0] == NULL){ linha = 2; coluna = 0;}
                    else if(velha[0][0] == 'O' && velha[2][0] == 'O' && velha[1][0] == NULL){ linha = 1; coluna = 0;}
                    else if(velha[1][0] == 'O' && velha[2][0] == 'O' && velha[0][0] == NULL){ linha = 0; coluna = 0;}

                    else if(velha[0][1] == 'O' && velha[1][1] == 'O' && velha[2][1] == NULL){ linha = 2; coluna = 1;}
                    else if(velha[0][1] == 'O' && velha[2][1] == 'O' && velha[1][1] == NULL){ linha = 1; coluna = 1;}
                    else if(velha[1][1] == 'O' && velha[2][1] == 'O' && velha[0][1] == NULL){ linha = 0; coluna = 1;}

                    else if(velha[0][2] == 'O' && velha[1][2] == 'O' && velha[2][2] == NULL){ linha = 2; coluna = 2;}
                    else if(velha[0][2] == 'O' && velha[2][2] == 'O' && velha[1][2] == NULL){ linha = 1; coluna = 2;}
                    else if(velha[1][2] == 'O' && velha[2][2] == 'O' && velha[0][2] == NULL){ linha = 0; coluna = 2;}

                    else if(velha[0][0] == 'O' && velha[1][1] == 'O' && velha[2][2] == NULL){ linha = 2; coluna = 2;}
                    else if(velha[0][0] == 'O' && velha[2][2] == 'O' && velha[1][1] == NULL){ linha = 1; coluna = 1;}
                    else if(velha[1][1] == 'O' && velha[2][2] == 'O' && velha[0][0] == NULL){ linha = 0; coluna = 0;}

                    else if(velha[2][0] == 'O' && velha[1][1] == 'O' && velha[0][2] == NULL){ linha = 0; coluna = 2;}
                    else if(velha[2][0] == 'O' && velha[0][2] == 'O' && velha[1][1] == NULL){ linha = 1; coluna = 1;}
                    else if(velha[1][1] == 'O' && velha[0][2] == 'O' && velha[2][0] == NULL){ linha = 2; coluna = 0;}


                    else{ //Pega um valor aleatorio entre 1 e 9 como se fosse uma escolha caso as condições acima não possam ser utilizadas
                    bole = 1;
                    while(bole){
                        aux = rand()%10;
                        if(aux > 0 && aux < 10){bole = 0;}
                        }
                    escolha = aux + 48;


                        if (escolha == '1'){linha = 0; coluna = 0;}
                        else if (escolha == '2'){linha = 0; coluna = 1;}
                        else if(escolha == '3'){linha = 0; coluna = 2;}
                        else  if(escolha == '4'){linha = 1; coluna = 0;}
                        else  if(escolha == '5'){linha = 1; coluna = 1;}
                        else   if(escolha == '6'){linha = 1; coluna = 2;}
                        else  if(escolha == '7'){linha = 2; coluna = 0;}
                        else  if(escolha == '8'){linha = 2; coluna = 1;}
                        else if(escolha == '9'){linha = 2; coluna = 2;}
                    }

                if(velha[linha][coluna] == 'X' || velha[linha][coluna] == 'O'){
                    x_vs_pc_rounds_medio(velha, amostrador, nomeplayer, qual_round);}

                else{
                    velha[linha][coluna] = 'O';
                    amostrador[linha][coluna] = NULL;
                    verificador_xvspc(velha, nomeplayer, qual_round);
                    qual_round++;
                    x_vs_pc_rounds_medio(velha, amostrador, nomeplayer, qual_round);}
              }



}
