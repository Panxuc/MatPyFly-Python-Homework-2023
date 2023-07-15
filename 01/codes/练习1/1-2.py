class Solution:
    def compressString(self, S: str) -> str:
        if len(S) <= 1:
            return S
        else:
            C = ""
            i = 0
            count = 1
            for j in range(1, len(S)):
                if j < len(S) - 1:
                    if S[j] == S[i]:
                        count += 1
                    else:
                        C = C + S[i] + str(count)
                        i = j
                        count = 1
                else:
                    if S[j] == S[i]:
                        count += 1
                        C = C + S[i] + str(count)
                    else:
                        C = C + S[i] + str(count) + S[j] + "1"
            if len(C) < len(S):
                return C
            else:
                return S
